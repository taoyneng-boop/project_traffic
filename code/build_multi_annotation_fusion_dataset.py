#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build fused text-label datasets from multiple annotation spreadsheets.

Fusion modes:
- concat: concatenate all annotator descriptions.
- normalized: clean, deduplicate, and concatenate annotator descriptions.
- structured: combine environment/category metadata with normalized annotator evidence.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_ANNOTATION_FILES = [
    Path(r"C:/Users/Administrator/WPSDrive/1076947574/WPS云盘/小学期数据标注.xlsx"),
    Path(r"C:/Users/Administrator/WPSDrive/1076947574/WPS云盘/小学期数据标注(1).xlsx"),
    Path(r"C:/Users/Administrator/WPSDrive/1076947574/WPS云盘/小学期数据标注(2).xlsx"),
    Path(r"C:/Users/Administrator/WPSDrive/1076947574/WPS云盘/小学期数据标注(3).xlsx"),
    Path(r"C:/Users/Administrator/WPSDrive/1076947574/WPS云盘/小学期数据标注(4).xlsx"),
    Path(r"C:/Users/Administrator/WPSDrive/1076947574/WPS云盘/小学期数据标注(5).xlsx"),
]


def video_id(value) -> str:
    text = "" if value is None else str(value).strip()
    text = text.strip("'\" ")
    if text.endswith(".avi"):
        text = text[:-4]
    if text.endswith(".0"):
        text = text[:-2]
    try:
        text = str(int(float(text)))
    except Exception:
        pass
    digits = re.findall(r"\d+", text)
    if digits:
        text = digits[0]
    return text.zfill(5)


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    text = text.strip("“”\"' ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_base_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_annotation_file(path: Path, annotator_index: int) -> Dict[str, str]:
    import pandas as pd

    xl = pd.ExcelFile(path)
    df = pd.read_excel(path, sheet_name=xl.sheet_names[0])
    cols = list(df.columns)
    mapping: Dict[str, str] = {}

    for _, row in df.iterrows():
        row_dict = {str(col): row[col] for col in cols}
        vid = ""
        text_parts: List[str] = []

        if "video_id" in row_dict:
            vid = video_id(row_dict.get("video_id"))
            text_parts.append(clean_text(row_dict.get("original")))
            text_parts.append(clean_text(row_dict.get("note")))
        elif "video_name" in row_dict:
            vid = video_id(row_dict.get("video_name"))
            text_parts.append(clean_text(row_dict.get("course_accident_analysis")))
        elif "filename" in row_dict:
            vid = video_id(row_dict.get("filename"))
            if "entities" in row_dict or "actions" in row_dict or "result" in row_dict:
                text_parts.extend(
                    [
                        clean_text(row_dict.get("entities")),
                        clean_text(row_dict.get("actions")),
                        clean_text(row_dict.get("result")),
                    ]
                )
            else:
                text_parts.append(clean_text(row_dict.get("Unnamed: 1")))
        elif "Video Id" in row_dict:
            vid = video_id(row_dict.get("Video Id"))
            text_parts.append(clean_text(row_dict.get("Label")))

        text = " ".join(part for part in text_parts if part)
        if vid and text:
            mapping[vid] = text
    print(f"[标注] annotator_{annotator_index}: {path.name}, 有效={len(mapping)}")
    return mapping


def dedupe_texts(texts: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for text in texts:
        cleaned = clean_text(text)
        key = cleaned.lower().rstrip(".")
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def fused_text(row: Dict[str, str], annotations: List[str], mode: str) -> str:
    annotations = dedupe_texts(annotations)
    if mode == "concat":
        parts = [row["text_label"]] + annotations
        return " ".join(part for part in parts if part)
    if mode == "normalized":
        parts = annotations or [row["text_label"]]
        return " ".join(parts)
    if mode == "structured":
        evidence = " ".join(annotations) if annotations else row["text_label"]
        scene = row.get("scene", "unknown scene").lower()
        light = row.get("light_conditions", "unknown light").lower()
        weather = row.get("weather", "unknown weather").lower()
        category = row.get("category_en", "unknown collision conflict")
        return (
            f"Traffic accident risk scene. Environment: {scene} road, {light} light, {weather} weather. "
            f"Risk category: {category}. Multi-annotator evidence: {evidence}"
        )
    raise ValueError(mode)


def write_csv(rows: List[Dict[str, str]], path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[输出] {path} rows={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-annotation fused CLIP datasets.")
    parser.add_argument("--base-csv", type=Path, default=Path("label_data/meta0713_clip_lora_dataset_fixed120.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("label_data/multi_annotation_fusion"))
    parser.add_argument("--annotation-files", nargs="*", type=Path, default=DEFAULT_ANNOTATION_FILES)
    args = parser.parse_args()

    base_rows = read_base_csv(args.base_csv)
    annotation_maps = [load_annotation_file(path, idx + 1) for idx, path in enumerate(args.annotation_files)]

    fieldnames = list(base_rows[0].keys()) + [
        "annotation_count",
        "annotator_texts",
        "fusion_mode",
    ]
    summaries = []
    for mode in ["concat", "normalized", "structured"]:
        output_rows: List[Dict[str, str]] = []
        for row in base_rows:
            vid = video_id(row["video_name"])
            texts = [mapping[vid] for mapping in annotation_maps if vid in mapping and mapping[vid]]
            new_row = dict(row)
            new_row["text_label"] = fused_text(row, texts, mode)
            new_row["annotation_count"] = str(len(texts))
            new_row["annotator_texts"] = " || ".join(dedupe_texts(texts))
            new_row["fusion_mode"] = mode
            output_rows.append(new_row)
        output_path = args.output_dir / f"meta0713_fixed120_{mode}.csv"
        write_csv(output_rows, output_path, fieldnames)
        summaries.append(
            {
                "mode": mode,
                "path": str(output_path),
                "rows": len(output_rows),
                "avg_annotation_count": sum(int(row["annotation_count"]) for row in output_rows) / max(len(output_rows), 1),
            }
        )

    summary_path = args.output_dir / "fusion_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "path", "rows", "avg_annotation_count"])
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[汇总] {summary_path}")


if __name__ == "__main__":
    main()
