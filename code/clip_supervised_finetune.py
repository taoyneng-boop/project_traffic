#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
英文规范文本 + 分类监督的 CLIP 微调实验。

依赖安装：
pip install torch torchvision pillow ftfy regex tqdm pandas openpyxl
pip install git+https://github.com/openai/CLIP.git

支持训练模式：
1. image：只微调 CLIP 视觉编码器和分类头；
2. full：微调整个 CLIP 和分类头；
3. lora：冻结 CLIP 主体，在视觉 Transformer MLP 中加入 LoRA 低秩适配器，并训练分类头。

输出：
metrics.json、per_sample_test.csv、finetune_report.md。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def 打印(message: str) -> None:
    print(message, flush=True)


def 检查依赖():
    try:
        import torch  # type: ignore
        import clip  # type: ignore
        from PIL import Image  # type: ignore
        from torch import nn  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "缺少依赖，请先安装：\n"
            "pip install torch torchvision pillow ftfy regex tqdm pandas openpyxl\n"
            "pip install git+https://github.com/openai/CLIP.git"
        ) from exc
    return torch, clip, Image, nn, DataLoader, Dataset


def 读取数据(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frames = [p for p in (row.get("frame_paths") or "").split("|") if p.strip()]
            text = (row.get("text_label") or "").strip()
            label = 标签编号(row)
            if frames and text and label:
                row["risk_label_id"] = label
                rows.append(dict(row))
    return rows


def 标签编号(row: Dict[str, str]) -> str:
    label = (row.get("risk_label_id") or "").strip()
    if label:
        return str(int(float(label)))
    category = (row.get("source_category") or "").strip()
    if category:
        return str(int(float(category)) - 1)
    return ""


def 划分数据(rows: List[Dict[str, str]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    copied = list(rows)
    random.Random(seed).shuffle(copied)
    split = max(1, min(len(copied) - 1, int(len(copied) * train_ratio)))
    return copied[:split], copied[split:]


def 类别数量(rows: Sequence[Dict[str, str]]) -> int:
    return max(int(row["risk_label_id"]) for row in rows) + 1


def 计算类别权重(rows: Sequence[Dict[str, str]], num_classes: int, torch, device: str, mode: str):
    if mode == "none":
        return None, [1.0 for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]
    for row in rows:
        label = int(row["risk_label_id"])
        if 0 <= label < num_classes:
            counts[label] += 1
    safe_counts = [max(c, 1) for c in counts]
    total = float(sum(safe_counts))
    if mode == "balanced":
        weights = [total / (num_classes * c) for c in safe_counts]
    elif mode == "sqrt":
        weights = [(total / (num_classes * c)) ** 0.5 for c in safe_counts]
    else:
        raise ValueError(mode)
    mean_weight = sum(weights) / len(weights)
    weights = [w / mean_weight for w in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device), weights


def 生成训练序列(train_rows: Sequence[Dict[str, str]], class_weight_values, sampling_mode: str):
    epoch_rows = list(train_rows)
    if sampling_mode == "none":
        random.shuffle(epoch_rows)
        return epoch_rows
    if sampling_mode == "weighted":
        weights = [class_weight_values[int(row["risk_label_id"])] for row in train_rows]
        return random.choices(list(train_rows), weights=weights, k=len(train_rows))
    raise ValueError(sampling_mode)


def 读取多帧图像(row: Dict[str, str], preprocess, Image, torch, device: str):
    tensors = []
    for frame_path in row["frame_paths"].split("|"):
        frame_path = frame_path.strip()
        if not frame_path:
            continue
        image = Image.open(frame_path).convert("RGB")
        tensors.append(preprocess(image))
    if not tensors:
        raise RuntimeError(f"样本缺少可读取帧：{row.get('video_name')}")
    return torch.stack(tensors, dim=0).to(device)


def 批量读取多帧图像(rows, preprocess, Image, torch, device: str):
    videos = []
    for row in rows:
        videos.append(读取多帧图像(row, preprocess, Image, torch, device))
    return torch.stack(videos, dim=0)


def 编码图像(row: Dict[str, str], model, preprocess, Image, torch, device: str):
    frames = 读取多帧图像(row, preprocess, Image, torch, device)
    features = model.encode_image(frames)
    features = features.float().mean(dim=0)
    return features


def 批量视觉特征(rows, model, preprocess, Image, torch, device, video_batch_size: int = 16):
    all_features = []
    for start in range(0, len(rows), video_batch_size):
        batch_rows = rows[start : start + video_batch_size]
        videos = 批量读取多帧图像(batch_rows, preprocess, Image, torch, device)
        batch_size, frame_count = videos.shape[:2]
        flat_frames = videos.reshape(batch_size * frame_count, *videos.shape[2:])
        flat_features = model.encode_image(flat_frames).float()
        frame_features = flat_features.reshape(batch_size, frame_count, -1)
        all_features.append(frame_features.mean(dim=1))
    feats = torch.cat(all_features, dim=0)
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def 批量文本特征(texts, model, clip, torch, device):
    tokens = clip.tokenize(texts, truncate=True).to(device)
    feats = model.encode_text(tokens).float()
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def 检索指标(similarity, torch) -> Dict[str, float]:
    n = similarity.shape[0]
    ranks: List[int] = []
    margins: List[float] = []
    true_values: List[float] = []
    mismatch_values: List[float] = []
    for i in range(n):
        row = similarity[i]
        order = torch.argsort(row, descending=True)
        rank = int((order == i).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)
        true_score = float(row[i].item())
        wrong = torch.cat([row[:i], row[i + 1 :]]) if n > 1 else row
        best_wrong = float(wrong.max().item()) if n > 1 else true_score
        true_values.append(true_score)
        mismatch_values.append(float(wrong.mean().item()) if n > 1 else true_score)
        margins.append(true_score - best_wrong)
    return {
        "sample_count": float(n),
        "mean_true_similarity": sum(true_values) / n,
        "mean_mismatch_similarity": sum(mismatch_values) / n,
        "mean_margin_vs_best_wrong": sum(margins) / n,
        "positive_best_wrong_margin_ratio": sum(1 for m in margins if m > 0) / n,
        "recall_at_1": sum(1 for r in ranks if r <= 1) / n,
        "recall_at_5": sum(1 for r in ranks if r <= 5) / n,
        "recall_at_10": sum(1 for r in ranks if r <= 10) / n,
        "mean_rank": sum(ranks) / n,
        "median_rank": float(sorted(ranks)[n // 2]),
    }


def 分类指标(logits, labels, torch) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    correct = (pred == labels).float()
    metrics = {
        "classification_accuracy": float(correct.mean().item()),
    }
    num_classes = int(labels.max().item()) + 1 if labels.numel() else 0
    for cls_id in range(num_classes):
        mask = labels == cls_id
        if bool(mask.any().item()):
            metrics[f"class_{cls_id}_accuracy"] = float((pred[mask] == labels[mask]).float().mean().item())
            metrics[f"class_{cls_id}_count"] = float(mask.sum().item())
    return metrics


def 评估(rows, model, classifier, preprocess, clip, Image, torch, device, batch_size: int):
    model.eval()
    classifier.eval()
    texts = [row["text_label"] for row in rows]
    labels = torch.tensor([int(row["risk_label_id"]) for row in rows], dtype=torch.long, device=device)
    with torch.no_grad():
        image_features = 批量视觉特征(rows, model, preprocess, Image, torch, device, batch_size)
        text_features = 批量文本特征(texts, model, clip, torch, device)
        similarity = image_features @ text_features.T
        retrieval = 检索指标(similarity, torch)
        logits = classifier(image_features)
        cls = 分类指标(logits, labels, torch)
    return {**retrieval, **cls}, similarity


def 写逐样本(path: Path, rows, similarity, torch) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_name", "risk_type", "true_rank", "true_similarity", "top1_video", "top1_similarity", "text_label"],
        )
        writer.writeheader()
        for i, row in enumerate(rows):
            sims = similarity[i]
            order = torch.argsort(sims, descending=True)
            rank = int((order == i).nonzero(as_tuple=False)[0].item()) + 1
            top1 = int(order[0].item())
            writer.writerow(
                {
                    "video_name": row["video_name"],
                    "risk_type": row.get("risk_type", ""),
                    "true_rank": rank,
                    "true_similarity": f"{float(sims[i].item()):.6f}",
                    "top1_video": rows[top1]["video_name"],
                    "top1_similarity": f"{float(sims[top1].item()):.6f}",
                    "text_label": row["text_label"],
                }
            )


class LoRALinear:
    """给已有 Linear 增加低秩增量，保留原始权重冻结。"""

    def __init__(self, torch, nn, base, rank: int = 4, alpha: float = 8.0):
        self.torch = torch
        self.nn = nn
        self.base = base
        self.rank = rank
        self.alpha = alpha

    def build(self):
        torch = self.torch
        nn = self.nn
        base = self.base

        class _LoRALinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base
                for param in self.base.parameters():
                    param.requires_grad = False
                self.lora_a = nn.Parameter(
                    torch.randn(
                        self.base.in_features,
                        self_rank,
                        dtype=self.base.weight.dtype,
                        device=self.base.weight.device,
                    )
                    * 0.01
                )
                self.lora_b = nn.Parameter(
                    torch.zeros(
                        self_rank,
                        self.base.out_features,
                        dtype=self.base.weight.dtype,
                        device=self.base.weight.device,
                    )
                )
                self.scaling = self_alpha / self_rank

            def forward(self, x):
                return self.base(x) + (x @ self.lora_a @ self.lora_b) * self.scaling

        self_rank = self.rank
        self_alpha = self.alpha
        return _LoRALinear()


def 注入_lora(model, torch, nn, rank: int, alpha: float) -> int:
    """在视觉 Transformer 的 MLP 线性层注入 LoRA。"""
    count = 0
    for block in model.visual.transformer.resblocks:
        block.mlp.c_fc = LoRALinear(torch, nn, block.mlp.c_fc, rank, alpha).build()
        block.mlp.c_proj = LoRALinear(torch, nn, block.mlp.c_proj, rank, alpha).build()
        count += 2
    return count


def 设置训练参数(model, classifier, train_mode: str, torch, nn, lora_rank: int, lora_alpha: float):
    for p in model.parameters():
        p.requires_grad = False
    injected = 0
    if train_mode == "image":
        for p in model.visual.parameters():
            p.requires_grad = True
        model.logit_scale.requires_grad = True
    elif train_mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif train_mode == "lora":
        injected = 注入_lora(model, torch, nn, lora_rank, lora_alpha)
        model.logit_scale.requires_grad = True
    else:
        raise ValueError(train_mode)
    for p in classifier.parameters():
        p.requires_grad = True
    params = [p for p in list(model.parameters()) + list(classifier.parameters()) if p.requires_grad]
    return params, injected


def 训练(
    train_rows,
    model,
    classifier,
    preprocess,
    clip,
    Image,
    torch,
    device,
    epochs: int,
    batch_size: int,
    lr: float,
    cls_weight: float,
    contrastive_weight: float,
    class_weights,
    class_weight_values,
    label_smoothing: float,
    sampling_mode: str,
):
    optimizer = torch.optim.AdamW([p for p in list(model.parameters()) + list(classifier.parameters()) if p.requires_grad], lr=lr, weight_decay=1e-4)
    cls_criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        classifier.train()
        epoch_rows = 生成训练序列(train_rows, class_weight_values, sampling_mode)
        total_loss = 0.0
        total_contrast = 0.0
        total_cls = 0.0
        steps = 0
        for start in range(0, len(epoch_rows), batch_size):
            batch = epoch_rows[start : start + batch_size]
            if len(batch) < 2:
                continue
            texts = [row["text_label"] for row in batch]
            labels = torch.tensor([int(row["risk_label_id"]) for row in batch], dtype=torch.long, device=device)
            image_features = 批量视觉特征(batch, model, preprocess, Image, torch, device, len(batch))
            text_features = 批量文本特征(texts, model, clip, torch, device)
            logits_per_image = model.logit_scale.exp() * image_features @ text_features.T
            targets = torch.arange(len(batch), device=device)
            contrast_loss = (
                torch.nn.functional.cross_entropy(logits_per_image, targets)
                + torch.nn.functional.cross_entropy(logits_per_image.T, targets)
            ) / 2
            cls_logits = classifier(image_features)
            cls_loss = cls_criterion(cls_logits, labels)
            loss = contrastive_weight * contrast_loss + cls_weight * cls_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_contrast += float(contrast_loss.item())
            total_cls += float(cls_loss.item())
            steps += 1
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(steps, 1),
                "contrastive_loss": total_contrast / max(steps, 1),
                "classification_loss": total_cls / max(steps, 1),
            }
        )
        打印(
            f"[训练] epoch={epoch}/{epochs}, loss={history[-1]['train_loss']:.4f}, "
            f"contrast={history[-1]['contrastive_loss']:.4f}, cls={history[-1]['classification_loss']:.4f}"
        )
    return history


def 写报告(path: Path, config: Dict[str, object], metrics: Dict[str, object]) -> None:
    before = metrics["before_finetune_test"]
    after = metrics["after_finetune_test"]
    lines = [
        "# CLIP 英文文本 + 分类监督微调实验",
        "",
        f"- 输入数据：`{config['input_csv']}`",
        f"- 训练模式：`{config['train_mode']}`",
        f"- 样本划分：{config['train_count']} / {config['test_count']}",
        f"- 分类损失权重：{config['cls_weight']}",
        f"- 对比损失权重：{config['contrastive_weight']}",
        f"- 类别权重模式：{config['class_weight_mode']}",
        f"- 训练采样模式：{config['sampling_mode']}",
        f"- 标签平滑：{config['label_smoothing']}",
        "",
        "| 阶段 | R@1 | R@5 | R@10 | Mean Rank | 分类准确率 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 微调前 | {before['recall_at_1']:.2%} | {before['recall_at_5']:.2%} | {before['recall_at_10']:.2%} | {before['mean_rank']:.2f} | {before['classification_accuracy']:.2%} |",
        f"| 微调后 | {after['recall_at_1']:.2%} | {after['recall_at_5']:.2%} | {after['recall_at_10']:.2%} | {after['mean_rank']:.2f} | {after['classification_accuracy']:.2%} |",
        "",
        "说明：R@k 和 Mean Rank 衡量图文检索；分类准确率衡量视频视觉特征对事故类型的识别能力。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP 英文文本 + 分类监督微调")
    parser.add_argument("--input-csv", type=Path, default=Path("label_data/english_norm_text_label.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("clip_output/english_supervised_norm_image"))
    parser.add_argument("--model", default="ViT-B/32")
    parser.add_argument("--train-mode", choices=["image", "full", "lora"], default="image")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--cls-weight", type=float, default=0.3)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--class-weight-mode", choices=["none", "balanced", "sqrt"], default="none")
    parser.add_argument("--sampling-mode", choices=["none", "weighted"], default="none")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch, clip, Image, nn, _DataLoader, _Dataset = 检查依赖()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    打印(f"[环境] model={args.model}, device={device}, train_mode={args.train_mode}")
    if device == "cuda":
        打印(f"[环境] GPU={torch.cuda.get_device_name(0)}")

    rows = 读取数据(args.input_csv)
    train_rows, test_rows = 划分数据(rows, args.train_ratio, args.seed)
    num_classes = 类别数量(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess = clip.load(args.model, device=device)
    model.float()
    classifier = nn.Linear(model.visual.output_dim, num_classes).to(device).float()
    params, injected = 设置训练参数(model, classifier, args.train_mode, torch, nn, args.lora_rank, args.lora_alpha)
    class_weights, class_weight_values = 计算类别权重(train_rows, num_classes, torch, device, args.class_weight_mode)
    打印(f"[数据] 总样本={len(rows)}, train={len(train_rows)}, test={len(test_rows)}, classes={num_classes}")
    打印(f"[训练参数] trainable={sum(p.numel() for p in params):,}, lora_layers={injected}")
    打印(f"[分类监督] cls_weight={args.cls_weight}, contrastive_weight={args.contrastive_weight}, class_weight_mode={args.class_weight_mode}, sampling_mode={args.sampling_mode}, label_smoothing={args.label_smoothing}")

    打印("[评估] 微调前")
    before, _ = 评估(test_rows, model, classifier, preprocess, clip, Image, torch, device, args.batch_size)
    打印(f"[微调前] R@1={before['recall_at_1']:.2%}, R@5={before['recall_at_5']:.2%}, Acc={before['classification_accuracy']:.2%}, MeanRank={before['mean_rank']:.2f}")

    history = 训练(
        train_rows,
        model,
        classifier,
        preprocess,
        clip,
        Image,
        torch,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.cls_weight,
        args.contrastive_weight,
        class_weights,
        class_weight_values,
        args.label_smoothing,
        args.sampling_mode,
    )

    打印("[评估] 微调后")
    after, similarity = 评估(test_rows, model, classifier, preprocess, clip, Image, torch, device, args.batch_size)
    打印(f"[微调后] R@1={after['recall_at_1']:.2%}, R@5={after['recall_at_5']:.2%}, Acc={after['classification_accuracy']:.2%}, MeanRank={after['mean_rank']:.2f}")

    metrics = {
        "config": {
            "input_csv": str(args.input_csv),
            "model": args.model,
            "train_mode": args.train_mode,
            "train_ratio": args.train_ratio,
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "cls_weight": args.cls_weight,
            "contrastive_weight": args.contrastive_weight,
            "class_weight_mode": args.class_weight_mode,
            "class_weight_values": class_weight_values,
            "sampling_mode": args.sampling_mode,
            "label_smoothing": args.label_smoothing,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "seed": args.seed,
            "num_classes": num_classes,
            "trainable_parameters": sum(p.numel() for p in params),
            "lora_layers": injected,
        },
        "before_finetune_test": before,
        "after_finetune_test": after,
        "train_history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    写逐样本(args.output_dir / "per_sample_test.csv", test_rows, similarity, torch)
    写报告(args.output_dir / "finetune_report.md", metrics["config"], metrics)
    打印(f"[完成] {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
