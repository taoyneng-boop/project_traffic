#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train CLIP ViT-B/16 + visual LoRA + Temporal Attention on the semantic-union
multi-annotation dataset.

Outputs:
- metrics.json with per-epoch train/test loss, Acc, R@1/Top1, R@5, MeanRank
- per_sample_test.csv
- loss_curve.png
- top1_acc_curve.png
- comparison_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def log(message: str) -> None:
    print(message, flush=True)


def load_deps():
    try:
        import torch  # type: ignore
        import clip  # type: ignore
        from PIL import Image  # type: ignore
        from torch import nn  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependencies: torch, clip, pillow") from exc
    return torch, clip, Image, nn


def label_id(row: Dict[str, str]) -> str:
    value = (row.get("risk_label_id") or "").strip()
    if value:
        return str(int(float(value)))
    category = (row.get("source_category") or "").strip()
    if category:
        return str(int(float(category)) - 1)
    return ""


def read_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frames = [part.strip() for part in (row.get("frame_paths") or "").split("|") if part.strip()]
            text = (row.get("text_label") or "").strip()
            label = label_id(row)
            if len(frames) == 5 and text and label:
                row = dict(row)
                row["risk_label_id"] = label
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No valid rows found: {path}")
    return rows


def split_rows(rows: List[Dict[str, str]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    copied = list(rows)
    random.Random(seed).shuffle(copied)
    split = max(1, min(len(copied) - 1, int(len(copied) * train_ratio)))
    return copied[:split], copied[split:]


def split_rows_kfold(rows: List[Dict[str, str]], folds: int, fold_index: int, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    if fold_index < 0 or fold_index >= folds:
        raise ValueError("--fold-index must be in [0, folds)")
    copied = list(rows)
    random.Random(seed).shuffle(copied)
    test_rows = [row for idx, row in enumerate(copied) if idx % folds == fold_index]
    train_rows = [row for idx, row in enumerate(copied) if idx % folds != fold_index]
    if not train_rows or not test_rows:
        raise RuntimeError("Empty train/test split from k-fold settings")
    return train_rows, test_rows


def num_classes(rows: Sequence[Dict[str, str]]) -> int:
    return max(int(row["risk_label_id"]) for row in rows) + 1


def normalize(features, torch):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class LoRALinear:
    def __init__(self, torch, nn, base, rank: int, alpha: float):
        self.torch = torch
        self.nn = nn
        self.base = base
        self.rank = rank
        self.alpha = alpha

    def build(self):
        torch = self.torch
        nn = self.nn
        base = self.base
        rank = self.rank
        alpha = self.alpha

        class _LoRALinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base
                for param in self.base.parameters():
                    param.requires_grad = False
                self.lora_a = nn.Parameter(
                    torch.randn(
                        self.base.in_features,
                        rank,
                        dtype=self.base.weight.dtype,
                        device=self.base.weight.device,
                    )
                    * 0.01
                )
                self.lora_b = nn.Parameter(
                    torch.zeros(
                        rank,
                        self.base.out_features,
                        dtype=self.base.weight.dtype,
                        device=self.base.weight.device,
                    )
                )
                self.scaling = alpha / rank

            def forward(self, x):
                return self.base(x) + (x @ self.lora_a @ self.lora_b) * self.scaling

        return _LoRALinear()


def inject_visual_lora(model, torch, nn, rank: int, alpha: float) -> int:
    count = 0
    for block in model.visual.transformer.resblocks:
        block.mlp.c_fc = LoRALinear(torch, nn, block.mlp.c_fc, rank, alpha).build()
        block.mlp.c_proj = LoRALinear(torch, nn, block.mlp.c_proj, rank, alpha).build()
        count += 2
    return count


def trainable_clip_state_dict(model):
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.detach().cpu()
    return state


def build_train_preprocess(preprocess, augment: str):
    if augment == "none":
        return preprocess
    try:
        from torchvision import transforms  # type: ignore
    except Exception as exc:
        log(f"[warn] torchvision transforms unavailable, disable image augment: {exc}")
        return preprocess
    if augment != "strong":
        raise ValueError(augment)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def apply_temporal_augmentation(frame_features, torch, drop_prob: float, max_drop: int, jitter_prob: float):
    batch_size, frame_count = frame_features.shape[:2]
    output = frame_features.clone()
    if max_drop > 0 and drop_prob > 0:
        for idx in range(batch_size):
            if random.random() < drop_prob:
                drop_count = random.randint(1, min(max_drop, max(frame_count - 1, 1)))
                keep = set(range(frame_count))
                candidates = list(range(frame_count - 1))
                random.shuffle(candidates)
                for drop_idx in candidates[:drop_count]:
                    keep.discard(drop_idx)
                kept = sorted(keep)
                for drop_idx in range(frame_count):
                    if drop_idx not in keep:
                        nearest = min(kept, key=lambda item: abs(item - drop_idx))
                        output[idx, drop_idx] = output[idx, nearest]
    if jitter_prob > 0:
        for idx in range(batch_size):
            if random.random() < jitter_prob and frame_count > 3:
                pos = random.randint(0, frame_count - 3)
                output[idx, pos], output[idx, pos + 1] = output[idx, pos + 1].clone(), output[idx, pos].clone()
    return output


def mixup_batch(video_features, logits, text_features, labels, torch, alpha: float):
    if alpha <= 0 or len(labels) < 2:
        return video_features, logits, text_features, labels, None
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    perm = torch.randperm(len(labels), device=labels.device)
    mixed_video = lam * video_features + (1 - lam) * video_features[perm]
    mixed_video = normalize(mixed_video, torch)
    mixed_logits = lam * logits + (1 - lam) * logits[perm]
    mixed_text = lam * text_features + (1 - lam) * text_features[perm]
    mixed_text = normalize(mixed_text, torch)
    return mixed_video, mixed_logits, mixed_text, labels, (perm, lam)


def hard_negative_margin_loss(similarity, labels, torch, margin: float, mode: str):
    if mode == "none" or similarity.shape[0] < 2:
        return similarity.new_tensor(0.0)
    n = similarity.shape[0]
    positive = similarity.diag()
    not_self = ~torch.eye(n, dtype=torch.bool, device=similarity.device)
    if mode == "same_class":
        same_class = labels.view(-1, 1) == labels.view(1, -1)
        hard_mask = same_class & not_self
        fallback_mask = not_self
        hard_mask = torch.where(hard_mask.any(dim=1, keepdim=True), hard_mask, fallback_mask)
    elif mode == "batch":
        hard_mask = not_self
    else:
        raise ValueError(mode)
    hardest_negative = similarity.masked_fill(~hard_mask, -1e9).max(dim=1).values
    return torch.relu(margin + hardest_negative - positive).mean()


def build_temporal_head(name: str, embed_dim: int, classes: int, torch, nn, dropout: float, transformer_layers: int):
    class TemporalAttentionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
            )
            self.proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))
            self.classifier = nn.Linear(embed_dim, classes)

        def forward(self, frame_features):
            weights = torch.softmax(self.score(frame_features).squeeze(-1), dim=1)
            pooled = (frame_features * weights.unsqueeze(-1)).sum(dim=1)
            video = normalize(self.proj(pooled), torch)
            logits = self.classifier(video)
            return video, logits, weights

    class TemporalTransformerHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos = nn.Parameter(torch.randn(1, 5, embed_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
            self.score = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
            self.proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))
            self.classifier = nn.Linear(embed_dim, classes)

        def forward(self, frame_features):
            encoded = self.encoder(frame_features + self.pos[:, : frame_features.shape[1], :])
            weights = torch.softmax(self.score(encoded).squeeze(-1), dim=1)
            pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
            video = normalize(self.proj(pooled), torch)
            logits = self.classifier(video)
            return video, logits, weights

    if name == "attention":
        return TemporalAttentionHead()
    if name == "transformer":
        return TemporalTransformerHead()
    raise ValueError(name)


def read_video_batch(rows, preprocess, Image, torch, device: str):
    videos = []
    for row in rows:
        cached = row.get("_video_tensor")
        if cached is not None:
            videos.append(cached)
            continue
        frames = []
        for frame_path in row["frame_paths"].split("|"):
            image = Image.open(frame_path.strip()).convert("RGB")
            frames.append(preprocess(image))
        videos.append(torch.stack(frames, dim=0))
    return torch.stack(videos, dim=0).to(device)


def encode_frame_features(rows, clip_model, preprocess, Image, torch, device: str, temporal_augment: bool = False, args=None):
    videos = read_video_batch(rows, preprocess, Image, torch, device)
    batch_size, frame_count = videos.shape[:2]
    flat = videos.reshape(batch_size * frame_count, *videos.shape[2:])
    flat_features = clip_model.encode_image(flat).float()
    frame_features = flat_features.reshape(batch_size, frame_count, -1)
    frame_features = normalize(frame_features, torch)
    if temporal_augment and args is not None:
        frame_features = apply_temporal_augmentation(
            frame_features,
            torch,
            args.frame_drop_prob,
            args.frame_drop_max,
            args.frame_jitter_prob,
        )
    return frame_features


def encode_text_features(texts: Sequence[str], clip_model, clip, torch, device: str):
    tokens = clip.tokenize(list(texts), truncate=True).to(device)
    features = clip_model.encode_text(tokens).float()
    return normalize(features, torch)


def class_weights(rows: Sequence[Dict[str, str]], classes: int, torch, device: str, mode: str):
    if mode == "none":
        return None, [1.0 for _ in range(classes)]
    counts = [0 for _ in range(classes)]
    for row in rows:
        counts[int(row["risk_label_id"])] += 1
    safe = [max(count, 1) for count in counts]
    total = float(sum(safe))
    if mode == "balanced":
        weights = [total / (classes * count) for count in safe]
    elif mode == "sqrt":
        weights = [(total / (classes * count)) ** 0.5 for count in safe]
    else:
        raise ValueError(mode)
    mean = sum(weights) / len(weights)
    weights = [weight / mean for weight in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device), weights


def epoch_batches(rows: Sequence[Dict[str, str]], batch_size: int, rng: random.Random, sample_weights: Sequence[float] | None):
    if sample_weights is None:
        epoch_rows = list(rows)
        rng.shuffle(epoch_rows)
    else:
        epoch_rows = rng.choices(list(rows), weights=sample_weights, k=len(rows))
    for start in range(0, len(epoch_rows), batch_size):
        batch = epoch_rows[start : start + batch_size]
        if len(batch) >= 2:
            yield batch


def retrieval_metrics(similarity, torch) -> Dict[str, float]:
    n = similarity.shape[0]
    ranks = []
    for idx in range(n):
        order = torch.argsort(similarity[idx], descending=True)
        ranks.append(int((order == idx).nonzero(as_tuple=False)[0].item()) + 1)
    return {
        "top1": sum(1 for rank in ranks if rank <= 1) / n,
        "recall_at_1": sum(1 for rank in ranks if rank <= 1) / n,
        "recall_at_5": sum(1 for rank in ranks if rank <= 5) / n,
        "recall_at_10": sum(1 for rank in ranks if rank <= 10) / n,
        "mean_rank": sum(ranks) / n,
    }


def rerank_similarity(similarity, logits, labels, torch, alpha: float, top_k: int):
    if alpha <= 0:
        return similarity
    top_k = max(1, min(int(top_k), similarity.shape[1]))
    class_probs = torch.softmax(logits, dim=1)
    candidate_labels = labels.view(1, -1).expand(similarity.shape[0], -1)
    label_scores = class_probs.gather(1, candidate_labels)
    blended = (1 - alpha) * similarity + alpha * label_scores
    reranked = similarity.clone()
    top_indices = torch.topk(similarity, k=top_k, dim=1).indices
    top_scores = torch.gather(blended, 1, top_indices)
    floor = similarity.min(dim=1, keepdim=True).values - 1.0
    reranked = floor.expand_as(similarity).clone()
    reranked.scatter_(1, top_indices, top_scores)
    return reranked


def classification_metrics(logits, labels, torch) -> Dict[str, float]:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float()
    return {"classification_accuracy": float(correct.mean().item())}


def evaluate(
    rows,
    clip_model,
    temporal_head,
    preprocess,
    clip,
    Image,
    torch,
    device: str,
    batch_size: int,
    rerank_alpha: float = 0.0,
    rerank_top_k: int = 10,
):
    clip_model.eval()
    temporal_head.eval()
    all_video = []
    all_text = []
    all_logits = []
    all_labels = []
    all_names = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            frame_features = encode_frame_features(batch, clip_model, preprocess, Image, torch, device)
            video_features, logits, _weights = temporal_head(frame_features)
            text_features = encode_text_features([row["text_label"] for row in batch], clip_model, clip, torch, device)
            all_video.append(video_features)
            all_text.append(text_features)
            all_logits.append(logits)
            all_labels.append(torch.tensor([int(row["risk_label_id"]) for row in batch], dtype=torch.long, device=device))
            all_names.extend([row["video_name"] for row in batch])
    video = torch.cat(all_video, dim=0)
    text = torch.cat(all_text, dim=0)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    similarity = video @ text.T
    reranked = rerank_similarity(similarity, logits, labels, torch, rerank_alpha, rerank_top_k)
    metrics = {**retrieval_metrics(similarity, torch), **classification_metrics(logits, labels, torch)}
    if rerank_alpha > 0:
        reranked_metrics = retrieval_metrics(reranked, torch)
        metrics.update(
            {
                "rerank_top1": reranked_metrics["top1"],
                "rerank_r5": reranked_metrics["recall_at_5"],
                "rerank_mean_rank": reranked_metrics["mean_rank"],
            }
        )
    return metrics, reranked, logits, labels, all_names


def train(
    train_rows,
    test_rows,
    clip_model,
    temporal_head,
    preprocess,
    clip,
    Image,
    torch,
    device: str,
    args,
    class_weight_tensor,
    class_weight_values,
    train_preprocess,
    eval_preprocess,
    output_dir: Path,
):
    params = [param for param in list(clip_model.parameters()) + list(temporal_head.parameters()) if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=args.label_smoothing)
    use_amp = bool(getattr(args, "amp", False)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_steps = max(1, math.ceil(len(train_rows) / args.batch_size) * args.epochs)
    warmup_steps = max(0, math.ceil(len(train_rows) / args.batch_size) * args.warmup_epochs)
    step_count = 0
    rng = random.Random(args.seed)
    sample_weights = None
    if args.sampling_mode == "weighted":
        sample_weights = [class_weight_values[int(row["risk_label_id"])] for row in train_rows]

    history = []
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        clip_model.train()
        temporal_head.train()
        total_loss = 0.0
        total_contrast = 0.0
        total_cls = 0.0
        total_hard = 0.0
        train_correct = 0
        train_seen = 0
        steps = 0
        for batch in epoch_batches(train_rows, args.batch_size, rng, sample_weights):
            labels = torch.tensor([int(row["risk_label_id"]) for row in batch], dtype=torch.long, device=device)
            texts = [row["text_label"] for row in batch]
            with torch.cuda.amp.autocast(enabled=use_amp):
                frame_features = encode_frame_features(batch, clip_model, train_preprocess, Image, torch, device, True, args)
                video_features, logits, _weights = temporal_head(frame_features)
                text_features = encode_text_features(texts, clip_model, clip, torch, device)
                video_features, logits, text_features, labels, mixup_info = mixup_batch(
                    video_features,
                    logits,
                    text_features,
                    labels,
                    torch,
                    args.mixup_alpha,
                )

                sim = clip_model.logit_scale.exp() * video_features @ text_features.T
                targets = torch.arange(len(batch), device=device)
                contrast_loss = (
                    torch.nn.functional.cross_entropy(sim, targets)
                    + torch.nn.functional.cross_entropy(sim.T, targets)
                ) / 2
                hard_loss = hard_negative_margin_loss(sim, labels, torch, args.hard_margin, args.hard_negative_mode)
                if mixup_info is None:
                    cls_loss = criterion(logits, labels)
                else:
                    perm, lam = mixup_info
                    cls_loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm])
                loss = (
                    args.contrastive_weight * contrast_loss
                    + args.cls_weight * cls_loss
                    + args.hard_weight * hard_loss
                )
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            step_count += 1
            if args.scheduler != "none":
                if step_count <= warmup_steps and warmup_steps > 0:
                    scale = step_count / warmup_steps
                else:
                    denom = max(total_steps - warmup_steps, 1)
                    progress = min(max((step_count - warmup_steps) / denom, 0.0), 1.0)
                    scale = 0.5 * (1.0 + math.cos(math.pi * progress)) if args.scheduler == "cosine" else 1.0
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * scale

            total_loss += float(loss.item())
            total_contrast += float(contrast_loss.item())
            total_cls += float(cls_loss.item())
            total_hard += float(hard_loss.item())
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())
            train_seen += len(batch)
            steps += 1
            if args.log_interval > 0 and steps % args.log_interval == 0:
                log(
                    f"[epoch {epoch:02d}/{args.epochs} step {steps:03d}] "
                    f"loss={total_loss / max(steps, 1):.4f} "
                    f"train_acc={train_correct / max(train_seen, 1):.2%}"
                )

        test_metrics = evaluate(
            test_rows,
            clip_model,
            temporal_head,
            eval_preprocess,
            clip,
            Image,
            torch,
            device,
            args.eval_batch_size,
            args.rerank_alpha,
            args.rerank_top_k,
        )[0]
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "contrastive_loss": total_contrast / max(steps, 1),
            "classification_loss": total_cls / max(steps, 1),
            "hard_negative_loss": total_hard / max(steps, 1),
            "batch_train_acc": train_correct / max(train_seen, 1),
            "train_acc": train_correct / max(train_seen, 1),
            "train_top1": "",
            "test_acc": test_metrics["classification_accuracy"],
            "test_top1": test_metrics["top1"],
            "test_r5": test_metrics["recall_at_5"],
            "test_mean_rank": test_metrics["mean_rank"],
            "rerank_top1": test_metrics.get("rerank_top1", ""),
            "rerank_r5": test_metrics.get("rerank_r5", ""),
            "lr": optimizer.param_groups[0]["lr"],
            "generalization_gap": train_correct / max(train_seen, 1) - test_metrics["classification_accuracy"],
        }
        history.append(row)
        log(
            f"[epoch {epoch:02d}/{args.epochs}] loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.2%} test_acc={row['test_acc']:.2%} "
            f"top1={row['test_top1']:.2%} r5={row['test_r5']:.2%} gap={row['generalization_gap']:.2%}"
        )
        monitor = row["test_top1"] if args.early_stop_metric == "top1" else row["test_acc"]
        if monitor > best_score:
            best_score = monitor
            best_epoch = epoch
            stale_epochs = 0
            if args.save_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "clip_trainable": trainable_clip_state_dict(clip_model),
                        "temporal_head": temporal_head.state_dict(),
                        "args": vars(args),
                        "metrics": row,
                    },
                    output_dir / "best_checkpoint.pt",
                )
        else:
            stale_epochs += 1
        write_history_csv(output_dir / "epoch_history_partial.csv", history)
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            log(f"[early_stop] metric={args.early_stop_metric}, best_epoch={best_epoch}, patience={args.early_stop_patience}")
            break
    return history


def write_per_sample(path: Path, rows, similarity, logits, labels, names, torch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = logits.argmax(dim=1).cpu().tolist()
    labels_list = labels.cpu().tolist()
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_name", "label", "pred", "true_rank", "true_similarity", "top1_video", "top1_similarity"],
        )
        writer.writeheader()
        for idx, name in enumerate(names):
            sims = similarity[idx]
            order = torch.argsort(sims, descending=True)
            rank = int((order == idx).nonzero(as_tuple=False)[0].item()) + 1
            top1 = int(order[0].item())
            writer.writerow(
                {
                    "video_name": name,
                    "label": int(labels_list[idx]),
                    "pred": int(preds[idx]),
                    "true_rank": rank,
                    "true_similarity": f"{float(sims[idx].item()):.6f}",
                    "top1_video": names[top1],
                    "top1_similarity": f"{float(sims[top1].item()):.6f}",
                }
            )


def plot_curves(output_dir: Path, history: Sequence[Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        log(f"[warn] matplotlib unavailable, using Pillow fallback: {exc}")
        plot_curves_with_pillow(output_dir, history)
        return

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["contrastive_loss"] for row in history], label="contrastive loss")
    plt.plot(epochs, [row["classification_loss"] for row in history], label="classification loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_acc"] for row in history], label="train Acc")
    plt.plot(epochs, [row["test_acc"] for row in history], label="test Acc")
    plt.plot(epochs, [row["test_top1"] for row in history], label="test Top1/R@1")
    plt.plot(epochs, [row["generalization_gap"] for row in history], label="Acc gap")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Top1 / Acc / Overfitting Trend")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "top1_acc_overfit_curve.png", dpi=160)
    plt.close()


def plot_curves_with_pillow(output_dir: Path, history: Sequence[Dict[str, float]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as exc:
        log(f"[warn] Pillow fallback plot failed: {exc}")
        return

    def draw_chart(path: Path, title: str, series: List[tuple[str, List[float], str]], y_label: str) -> None:
        width, height = 960, 600
        margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 80
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((margin_l, 24), title, fill="black", font=font)
        draw.line((margin_l, margin_t, margin_l, height - margin_b), fill="#333333", width=2)
        draw.line((margin_l, height - margin_b, width - margin_r, height - margin_b), fill="#333333", width=2)
        all_values = [value for _name, values, _color in series for value in values]
        min_y = min(0.0, min(all_values))
        max_y = max(all_values)
        if max_y == min_y:
            max_y = min_y + 1.0
        epochs = [float(row["epoch"]) for row in history]
        min_x = min(epochs)
        max_x = max(epochs)
        if min_x == max_x:
            max_x = min_x + 1

        def xy(epoch: float, value: float) -> tuple[float, float]:
            x = margin_l + (epoch - min_x) / (max_x - min_x) * (width - margin_l - margin_r)
            y = height - margin_b - (value - min_y) / (max_y - min_y) * (height - margin_t - margin_b)
            return x, y

        for tick in range(6):
            yv = min_y + (max_y - min_y) * tick / 5
            _x, y = xy(min_x, yv)
            draw.line((margin_l, y, width - margin_r, y), fill="#E5E7EB", width=1)
            draw.text((16, y - 7), f"{yv:.2f}", fill="#333333", font=font)
        draw.text((16, margin_t - 34), y_label, fill="#333333", font=font)
        draw.text((width - 110, height - margin_b + 28), "Epoch", fill="#333333", font=font)

        legend_x = margin_l
        for name, values, color in series:
            points = [xy(epoch, value) for epoch, value in zip(epochs, values)]
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            for point in points:
                x, y = point
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            draw.rectangle((legend_x, height - 34, legend_x + 14, height - 20), fill=color)
            draw.text((legend_x + 20, height - 36), name, fill="#111111", font=font)
            legend_x += 160
        image.save(path)

    epochs = [row["epoch"] for row in history]
    if not epochs:
        return
    draw_chart(
        output_dir / "loss_curve.png",
        "Training Loss Curves",
        [
            ("train loss", [float(row["train_loss"]) for row in history], "#2563EB"),
            ("contrastive", [float(row["contrastive_loss"]) for row in history], "#16A34A"),
            ("classification", [float(row["classification_loss"]) for row in history], "#DC2626"),
            ("hard negative", [float(row.get("hard_negative_loss", 0.0)) for row in history], "#9333EA"),
        ],
        "Loss",
    )
    draw_chart(
        output_dir / "top1_acc_overfit_curve.png",
        "Top1 / Acc / Overfitting Trend",
        [
            ("train Acc", [float(row["train_acc"]) for row in history], "#2563EB"),
            ("test Acc", [float(row["test_acc"]) for row in history], "#16A34A"),
            ("test Top1", [float(row["test_top1"]) for row in history], "#DC2626"),
            ("Acc gap", [float(row["generalization_gap"]) for row in history], "#9333EA"),
        ],
        "Score",
    )


def write_history_csv(path: Path, history: Sequence[Dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_comparison(path: Path, baseline: Dict[str, float], final_row: Dict[str, float], args) -> None:
    rows = [
        {
            "method": "baseline_vitb32_lora_ep12",
            "model": "ViT-B/32",
            "text_strategy": "fixed120 original",
            "temporal_head": "mean pooling",
            "epochs": 12,
            "top1": baseline["top1"],
            "acc": baseline["acc"],
            "r5": baseline["r5"],
            "mean_rank": baseline["mean_rank"],
            "overfit_gap": "",
        },
        {
            "method": path.parent.name,
            "model": args.model,
            "text_strategy": "semantic_union",
            "temporal_head": args.temporal_head,
            "epochs": args.epochs,
            "top1": final_row["test_top1"],
            "acc": final_row["test_acc"],
            "r5": final_row["test_r5"],
            "mean_rank": final_row["test_mean_rank"],
            "overfit_gap": final_row["generalization_gap"],
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic-union ViT-B/16 Temporal Attention LoRA training.")
    parser.add_argument("--input-csv", type=Path, default=Path("label_data/multi_annotation_fusion/meta0713_fixed120_compact_semantic_union.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("clip_output/compact_semantic_union_vitb16_temporal_attention_lora_ep20"))
    parser.add_argument("--model", default="ViT-B/16")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--folds", type=int, default=0)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cls-weight", type=float, default=0.6)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--hard-weight", type=float, default=0.0)
    parser.add_argument("--hard-margin", type=float, default=0.2)
    parser.add_argument("--hard-negative-mode", choices=["none", "batch", "same_class"], default="none")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-mode", choices=["none", "sqrt", "balanced"], default="sqrt")
    parser.add_argument("--sampling-mode", choices=["none", "weighted"], default="weighted")
    parser.add_argument("--image-augment", choices=["none", "strong"], default="none")
    parser.add_argument("--frame-drop-prob", type=float, default=0.0)
    parser.add_argument("--frame-drop-max", type=int, default=0)
    parser.add_argument("--frame-jitter-prob", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--temporal-head", choices=["attention", "transformer"], default="attention")
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-metric", choices=["top1", "acc"], default="top1")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--rerank-alpha", type=float, default=0.0)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    args = parser.parse_args()

    torch, clip, Image, nn = load_deps()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.input_csv)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.folds > 1:
        train_rows, test_rows = split_rows_kfold(rows, args.folds, args.fold_index, split_seed)
        split_note = f"{args.folds}-fold index={args.fold_index}"
    else:
        train_rows, test_rows = split_rows(rows, args.train_ratio, split_seed)
        split_note = f"random train_ratio={args.train_ratio}"
    classes = num_classes(rows)

    log(f"[env] model={args.model}, device={device}")
    if device == "cuda":
        log(f"[env] gpu={torch.cuda.get_device_name(0)}")
    log(f"[data] rows={len(rows)}, train={len(train_rows)}, test={len(test_rows)}, classes={classes}, split={split_note}")

    clip_model, preprocess = clip.load(args.model, device=device)
    clip_model.float()
    for param in clip_model.parameters():
        param.requires_grad = False
    lora_layers = inject_visual_lora(clip_model, torch, nn, args.lora_rank, args.lora_alpha)
    clip_model.logit_scale.requires_grad = True

    train_preprocess = build_train_preprocess(preprocess, args.image_augment)
    temporal_head = build_temporal_head(
        args.temporal_head,
        clip_model.visual.output_dim,
        classes,
        torch,
        nn,
        args.temporal_dropout,
        args.transformer_layers,
    ).to(device).float()
    weight_tensor, weight_values = class_weights(train_rows, classes, torch, device, args.class_weight_mode)
    trainable = sum(param.numel() for param in list(clip_model.parameters()) + list(temporal_head.parameters()) if param.requires_grad)
    log(
        f"[trainable] params={trainable:,}, lora_layers={lora_layers}, "
        f"head={args.temporal_head}, augment={args.image_augment}, dropout={args.temporal_dropout}, "
        f"label_smoothing={args.label_smoothing}, hard_negative={args.hard_negative_mode}, "
        f"hard_weight={args.hard_weight}, hard_margin={args.hard_margin}"
    )

    before, _sim, _logits, _labels, _names = evaluate(
        test_rows,
        clip_model,
        temporal_head,
        preprocess,
        clip,
        Image,
        torch,
        device,
        args.eval_batch_size,
        args.rerank_alpha,
        args.rerank_top_k,
    )
    log(f"[before] top1={before['top1']:.2%}, acc={before['classification_accuracy']:.2%}, r5={before['recall_at_5']:.2%}, mean_rank={before['mean_rank']:.2f}")

    history = train(
        train_rows,
        test_rows,
        clip_model,
        temporal_head,
        preprocess,
        clip,
        Image,
        torch,
        device,
        args,
        weight_tensor,
        weight_values,
        train_preprocess,
        preprocess,
        args.output_dir,
    )

    after, similarity, logits, labels, names = evaluate(
        test_rows,
        clip_model,
        temporal_head,
        preprocess,
        clip,
        Image,
        torch,
        device,
        args.eval_batch_size,
        args.rerank_alpha,
        args.rerank_top_k,
    )
    rerank_note = ""
    if args.rerank_alpha > 0:
        rerank_note = f", rerank_top1={after['rerank_top1']:.2%}, rerank_r5={after['rerank_r5']:.2%}"
    log(f"[after] top1={after['top1']:.2%}, acc={after['classification_accuracy']:.2%}, r5={after['recall_at_5']:.2%}, mean_rank={after['mean_rank']:.2f}{rerank_note}")

    metrics = {
        "config": vars(args) | {
            "device": device,
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "num_classes": classes,
            "trainable_parameters": trainable,
            "lora_layers": lora_layers,
            "class_weight_values": weight_values,
            "split_seed": split_seed,
            "split_note": split_note,
        },
        "before_test": before,
        "after_test": after,
        "train_history": history,
        "baseline": {
            "method": "CLIP ViT-B/32 LoRA fixed120 ep12",
            "top1": 0.11956521739130435,
            "acc": 0.42391306161880493,
            "r5": 0.3804347826086957,
            "mean_rank": 13.33695652173913,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_history_csv(args.output_dir / "epoch_history.csv", history)
    write_per_sample(args.output_dir / "per_sample_test.csv", test_rows, similarity, logits, labels, names, torch)
    plot_curves(args.output_dir, history)
    write_comparison(args.output_dir / "comparison_summary.csv", metrics["baseline"], history[-1], args)
    log(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
