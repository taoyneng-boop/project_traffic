#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a semantic-union text-label dataset from the existing multi-annotation CSV.

The previous normalized dataset removes exact duplicate annotation strings. This
script goes one step further: it splits annotator descriptions into sentences,
groups semantically similar sentences with CLIP text embeddings, keeps one
representative sentence per group, and writes a cleaner text_label.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def log(message: str) -> None:
    print(message, flush=True)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip("\"' ")
    if not text or text.lower() == "nan":
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def split_sentences(texts: Iterable[str]) -> List[str]:
    sentences: List[str] = []
    for text in texts:
        text = clean_text(text)
        if not text:
            continue
        parts = re.split(r"(?<=[.!?;])\s+", text)
        for part in parts:
            sentence = clean_text(part)
            if len(sentence) >= 8:
                sentences.append(sentence)
    return sentences


def strict_dedupe(sentences: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for sentence in sentences:
        key = sentence.lower().rstrip(".; ")
        if key and key not in seen:
            seen.add(key)
            output.append(sentence)
    return output


def lexical_similarity(a: str, b: str) -> float:
    tokenize = lambda x: set(re.findall(r"[a-zA-Z]+", x.lower()))
    left = tokenize(a)
    right = tokenize(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def choose_representative(cluster: Sequence[str]) -> str:
    risk_words = [
        "brake",
        "stop",
        "collision",
        "risk",
        "distance",
        "gap",
        "approach",
        "turn",
        "lane",
        "vehicle",
    ]

    def score(sentence: str) -> tuple[int, int]:
        lowered = sentence.lower()
        hits = sum(1 for word in risk_words if word in lowered)
        # Prefer content-rich but not overly long sentences.
        length_penalty = abs(len(sentence) - 110)
        return hits, -length_penalty

    return max(cluster, key=score)


def semantic_union_with_clip(sentences: List[str], model, clip, torch, threshold: float, device: str) -> tuple[List[str], List[List[str]]]:
    if not sentences:
        return [], []
    with torch.no_grad():
        tokens = clip.tokenize(sentences, truncate=True).to(device)
        features = model.encode_text(tokens).float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        similarity = (features @ features.T).cpu()

    clusters: List[List[int]] = []
    for idx, sentence in enumerate(sentences):
        placed = False
        for cluster in clusters:
            best_clip = max(float(similarity[idx, other].item()) for other in cluster)
            best_lex = max(lexical_similarity(sentence, sentences[other]) for other in cluster)
            if best_clip >= threshold or best_lex >= 0.55:
                cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])

    text_clusters = [[sentences[i] for i in cluster] for cluster in clusters]
    representatives = [choose_representative(cluster) for cluster in text_clusters]
    return representatives, text_clusters


def semantic_union_fallback(sentences: List[str]) -> tuple[List[str], List[List[str]]]:
    clusters: List[List[str]] = []
    for sentence in sentences:
        placed = False
        for cluster in clusters:
            if max(lexical_similarity(sentence, existing) for existing in cluster) >= 0.55:
                cluster.append(sentence)
                placed = True
                break
        if not placed:
            clusters.append([sentence])
    return [choose_representative(cluster) for cluster in clusters], clusters


def build_text(row: Dict[str, str], representatives: Sequence[str]) -> str:
    category = clean_text(row.get("category_en"))
    scene = clean_text(row.get("scene")).lower()
    light = clean_text(row.get("light_conditions")).lower()
    weather = clean_text(row.get("weather")).lower()
    prefix = []
    if scene or light or weather:
        prefix.append(f"Traffic scene: {scene or 'unknown'} road, {light or 'unknown'} light, {weather or 'unknown'} weather.")
    if category:
        prefix.append(f"Risk category: {category}.")
    evidence = list(representatives) or [clean_text(row.get("text_label"))]
    return " ".join(part for part in prefix + evidence if part)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic-union multi-annotation dataset.")
    parser.add_argument("--input-csv", type=Path, default=Path("label_data/multi_annotation_fusion/meta0713_fixed120_normalized.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("label_data/multi_annotation_fusion/meta0713_fixed120_semantic_union.csv"))
    parser.add_argument("--clip-model", default="ViT-B/32", help="Local CLIP text encoder used for semantic grouping.")
    parser.add_argument("--similarity-threshold", type=float, default=0.88)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError(f"No rows found: {args.input_csv}")

    try:
        import torch  # type: ignore

        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
    except Exception:
        device = "cpu"

    output_rows: List[Dict[str, str]] = []
    clip_ok = True
    torch = None
    clip = None
    model = None
    try:
        import torch as _torch  # type: ignore
        import clip as _clip  # type: ignore

        torch = _torch
        clip = _clip
        model, _ = clip.load(args.clip_model, device=device)
        model.eval()
        model.float()
        log(f"[clip] loaded {args.clip_model} on {device}")
    except Exception as exc:
        clip_ok = False
        log(f"[warn] CLIP text encoder unavailable, using lexical fallback: {exc}")

    for index, row in enumerate(rows, start=1):
        annotator_texts = [part for part in (row.get("annotator_texts") or "").split(" || ") if part.strip()]
        sentences = strict_dedupe(split_sentences(annotator_texts))
        try:
            if clip_ok:
                representatives, clusters = semantic_union_with_clip(sentences, model, clip, torch, args.similarity_threshold, device)
            else:
                representatives, clusters = semantic_union_fallback(sentences)
        except Exception as exc:
            clip_ok = False
            log(f"[warn] CLIP semantic grouping failed, using lexical fallback: {exc}")
            representatives, clusters = semantic_union_fallback(sentences)

        new_row = dict(row)
        new_row["text_label"] = build_text(row, representatives)
        new_row["fusion_mode"] = "semantic_union"
        new_row["semantic_group_count"] = str(len(clusters))
        new_row["semantic_union_sentences"] = " || ".join(representatives)
        output_rows.append(new_row)
        if index % 50 == 0 or index == len(rows):
            log(f"[semantic_union] {index}/{len(rows)}")

    fieldnames = list(rows[0].keys())
    for extra in ["semantic_group_count", "semantic_union_sentences"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    write_csv(args.output_csv, output_rows, fieldnames)
    log(f"[done] {args.output_csv} rows={len(output_rows)}")


if __name__ == "__main__":
    main()
