#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 meta0713.xls 构建 CLIP/LoRA 实验数据集。

依赖安装命令：
    .\\.venv_clip\\Scripts\\python.exe -m pip install xlrd

功能：
1. 读取 meta0713.xls 的字段信息；
2. 按 alert_frame_vip 到固定末帧 120 的事故前风险窗口平均抽取 5 帧；
   第 5 帧固定为第 120 帧，其余帧按窗口均匀抽取；如果窗口不足 5 帧，则向前补足；
3. 根据光照、天气、道路场景和事故类别编号生成英文 CLIP 文本；
4. 输出兼容 code/clip_supervised_finetune.py 的 CSV 数据集。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 原始视频目录。只读取，不写入。
DEFAULT_VIDEO_ROOT = Path(r"D:\2026小学期\train_positive_dlut_0713\train_positive_dlut_0713")
DEFAULT_ORIGINAL_META = DEFAULT_VIDEO_ROOT / "meta0713.xls"

# 将中文路径下的 xls 复制到项目内 ASCII 路径，避免部分 Windows 控制台编码导致读取失败。
DEFAULT_META_COPY = PROJECT_ROOT / "label_data" / "meta0713_source.xls"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "label_data" / "meta0713_clip_lora_dataset.csv"
DEFAULT_FIELD_REPORT = PROJECT_ROOT / "label_data" / "meta0713_field_report.json"
DEFAULT_FRAME_ROOT = PROJECT_ROOT / "frames" / "meta0713"
DEFAULT_FIXED_LAST_FRAME = 120


FIELD_MEANINGS = {
    "file_name": "视频编号，不带 .avi 后缀，例如 00822 对应 00822.avi",
    "light_conditions": "光照条件，例如 Normal、Bright 等",
    "weather": "天气条件，例如 Clear、Cloudy 等",
    "scene": "道路场景，例如 Urban、Sub-urban、Highway、Other",
    "alert_frame_vip": "人工标注的风险预警帧，表示事故前可观察风险开始出现的位置",
    "event_frame_vip": "事故发生帧，通常是碰撞或异常事件发生的位置",
    "category": "事故类别编号：1异方向-右，2异方向-左，3异方向-中，4同方向-右后，5同方向-右前，6同方向-左后，7同方向-左前，8同方向-中",
}


CATEGORY_MAP = {
    1: {
        "zh": "异方向-右",
        "risk_type": "opposite_direction_right_conflict",
        "en": "opposite-direction right-side collision conflict",
    },
    2: {
        "zh": "异方向-左",
        "risk_type": "opposite_direction_left_conflict",
        "en": "opposite-direction left-side collision conflict",
    },
    3: {
        "zh": "异方向-中",
        "risk_type": "opposite_direction_center_conflict",
        "en": "opposite-direction frontal center collision conflict",
    },
    4: {
        "zh": "同方向-右后",
        "risk_type": "same_direction_right_rear_conflict",
        "en": "same-direction right-rear collision conflict",
    },
    5: {
        "zh": "同方向-右前",
        "risk_type": "same_direction_right_front_conflict",
        "en": "same-direction right-front collision conflict",
    },
    6: {
        "zh": "同方向-左后",
        "risk_type": "same_direction_left_rear_conflict",
        "en": "same-direction left-rear collision conflict",
    },
    7: {
        "zh": "同方向-左前",
        "risk_type": "same_direction_left_front_conflict",
        "en": "same-direction left-front collision conflict",
    },
    8: {
        "zh": "同方向-中",
        "risk_type": "same_direction_center_conflict",
        "en": "same-direction center rear-end or forward collision conflict",
    },
}


def 打印(message: str) -> None:
    print(message, flush=True)


def 检查依赖():
    try:
        import xlrd  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "缺少 xlrd，无法读取老版 .xls。请执行："
            r".\.venv_clip\Scripts\python.exe -m pip install xlrd"
        ) from exc
    return xlrd


def 单元格转字符串(value) -> str:
    """把 Excel 单元格值转为稳定字符串，避免 00822 被变成 822.0。"""
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value).strip()


def 单元格转浮点(value) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number):
        return None
    return number


def 复制源表格(original_meta: Path, meta_copy: Path) -> None:
    """原文件不动；项目内只保存一个副本用于后续稳定读取。"""
    meta_copy.parent.mkdir(parents=True, exist_ok=True)
    if original_meta.exists():
        shutil.copy2(original_meta, meta_copy)
        打印(f"[表格] 已复制源表格到项目内：{meta_copy}")
    elif meta_copy.exists():
        打印(f"[表格] 使用已有项目内副本：{meta_copy}")
    else:
        raise FileNotFoundError(f"找不到 meta0713.xls：{original_meta}")


def 读取元数据(meta_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    xlrd = 检查依赖()
    workbook = xlrd.open_workbook(str(meta_path))
    sheet = workbook.sheet_by_index(0)
    headers = [单元格转字符串(sheet.cell_value(0, col)) for col in range(sheet.ncols)]

    rows: List[Dict[str, str]] = []
    for row_idx in range(1, sheet.nrows):
        row = {
            headers[col]: 单元格转字符串(sheet.cell_value(row_idx, col))
            for col in range(sheet.ncols)
        }
        if row.get("file_name"):
            rows.append(row)
    return rows, headers


def 写字段报告(rows: List[Dict[str, str]], headers: List[str], output_path: Path) -> None:
    category_counts: Dict[str, int] = {}
    for row in rows:
        category = row.get("category", "").strip()
        category_counts[category] = category_counts.get(category, 0) + 1

    report = {
        "row_count": len(rows),
        "columns": headers,
        "field_meanings": FIELD_MEANINGS,
        "category_counts": dict(sorted(category_counts.items(), key=lambda item: item[0])),
        "category_meanings": CATEGORY_MAP,
        "note": "category 已根据用户补充说明转换为中文类别和英文 CLIP 文本语义。",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    打印(f"[字段] 已写出字段说明：{output_path}")


def 视频编号(file_name: str) -> str:
    text = file_name.strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def ffprobe信息(video_path: Path) -> Tuple[float, float | None]:
    """返回 fps 和 duration。读取失败时给出保守默认值。"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        return 24.0, None
    info = json.loads(result.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        return 24.0, None

    stream = streams[0]
    fps_text = stream.get("avg_frame_rate") or "24/1"
    try:
        num, den = fps_text.split("/")
        fps = float(num) / max(float(den), 1.0)
    except Exception:
        fps = 24.0

    duration = 单元格转浮点(stream.get("duration"))
    return max(fps, 1.0), duration


def 均匀选择闭区间帧(start: int, end: int, frame_count: int) -> List[int]:
    """在闭区间 [start, end] 内按时序均匀选择 frame_count 帧。"""
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [end]

    return [
        int(round(start + idx * (end - start) / (frame_count - 1)))
        for idx in range(frame_count)
    ]


def 平均抽帧编号(row: Dict[str, str], video_path: Path, frame_count: int, fixed_last_frame: int | None) -> List[int]:
    """优先用事故前风险窗口；末帧可固定，窗口不足时向前补帧。返回 1 基语义帧号。"""
    alert = 单元格转浮点(row.get("alert_frame_vip"))
    event = 单元格转浮点(row.get("event_frame_vip"))

    if fixed_last_frame is not None:
        end = int(round(fixed_last_frame))
        start = int(round(alert)) if alert is not None else end - frame_count + 1
        if start > end or end - start + 1 < frame_count:
            start = end - frame_count + 1
        start = max(1, start)
        return 均匀选择闭区间帧(start, end, frame_count)

    if alert is not None and event is not None and event >= alert:
        start = int(round(alert))
        end = int(round(event))
        if end - start + 1 < frame_count:
            start = max(1, end - frame_count + 1)
        return 均匀选择闭区间帧(start, end, frame_count)

    fps, duration = ffprobe信息(video_path)
    if duration is None:
        return [idx for idx in range(1, frame_count + 1)]
    start_frame = max(0, int((duration - 3.5) * fps))
    end_frame = max(start_frame + frame_count, int(duration * fps) - 1)
    return [frame_index + 1 for frame_index in 均匀选择闭区间帧(start_frame, end_frame, frame_count)]


def 转为ffmpeg帧索引(frame_numbers: List[int]) -> List[int]:
    """把面向标注表的 1 基帧号转换为 ffmpeg select 使用的 0 基 n。"""
    return [max(0, number - 1) for number in frame_numbers]


def 使用ffmpeg抽帧(video_path: Path, frame_numbers: List[int], output_paths: List[Path]) -> bool:
    """一次 ffmpeg 调用抽取一个视频的 5 帧；失败时返回 False。"""
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    tmp_pattern = output_paths[0].parent / "_tmp_%02d.jpg"
    expression = "+".join(f"eq(n\\,{num})" for num in frame_numbers)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select='{expression}'",
        "-vsync",
        "0",
        "-q:v",
        "2",
        str(tmp_pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        return False

    tmp_files = sorted(output_paths[0].parent.glob("_tmp_*.jpg"))
    if len(tmp_files) < len(output_paths):
        for tmp_file in tmp_files:
            tmp_file.unlink(missing_ok=True)
        return False

    for tmp_file, final_path in zip(tmp_files, output_paths):
        if final_path.exists():
            final_path.unlink()
        tmp_file.replace(final_path)
    for tmp_file in tmp_files[len(output_paths) :]:
        tmp_file.unlink(missing_ok=True)
    return True


def 抽取帧(
    row: Dict[str, str],
    video_root: Path,
    frame_root: Path,
    frame_count: int,
    fixed_last_frame: int | None,
    overwrite: bool,
) -> Tuple[List[str], List[int]]:
    vid = 视频编号(row["file_name"])
    video_path = video_root / f"{vid}.avi"
    if not video_path.exists():
        raise FileNotFoundError(f"缺少视频：{video_path}")

    output_dir = frame_root / vid / "pre_event"
    output_paths = [output_dir / f"{vid}_pre_{idx:02d}.jpg" for idx in range(1, frame_count + 1)]
    frame_numbers = 平均抽帧编号(row, video_path, frame_count, fixed_last_frame)
    if not overwrite and all(path.exists() and path.stat().st_size > 0 for path in output_paths):
        return [str(path.resolve()) for path in output_paths], frame_numbers

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    ok = 使用ffmpeg抽帧(video_path, 转为ffmpeg帧索引(frame_numbers), output_paths)
    if not ok:
        raise RuntimeError(f"ffmpeg 抽帧失败：{video_path}，frame_numbers={frame_numbers}")
    return [str(path.resolve()) for path in output_paths], frame_numbers


def 英文文本(row: Dict[str, str]) -> str:
    """生成适合 CLIP 的简洁英文描述，保留可视觉识别的环境信息。"""
    light = row.get("light_conditions", "unknown light").strip() or "unknown light"
    weather = row.get("weather", "unknown weather").strip() or "unknown weather"
    scene = row.get("scene", "unknown road scene").strip() or "unknown road scene"
    category = row.get("category", "").strip()
    category_id = int(float(category)) if category else 0
    category_text = CATEGORY_MAP.get(category_id, {}).get("en", "unknown collision conflict")
    return (
        f"A pre-crash dashcam traffic scene on a {scene.lower()} road, "
        f"under {light.lower()} light and {weather.lower()} weather, "
        f"showing a {category_text}."
    )


def 构建数据集(
    rows: List[Dict[str, str]],
    video_root: Path,
    frame_root: Path,
    frame_count: int,
    fixed_last_frame: int | None,
    overwrite_frames: bool,
) -> List[Dict[str, str]]:
    output_rows: List[Dict[str, str]] = []
    skipped: List[str] = []

    if overwrite_frames and frame_root.exists():
        打印(f"[帧] 删除旧帧目录并重新生成：{frame_root}")
        shutil.rmtree(frame_root)

    for index, row in enumerate(rows, start=1):
        vid = 视频编号(row["file_name"])
        try:
            frame_paths, frame_numbers = 抽取帧(row, video_root, frame_root, frame_count, fixed_last_frame, overwrite_frames)
            category_number = int(float(row["category"]))
            category_info = CATEGORY_MAP.get(category_number, {})
            output_rows.append(
                {
                    "video_name": f"{vid}.avi",
                    "frame_paths": "|".join(frame_paths),
                    "frame_numbers": "|".join(str(number) for number in frame_numbers),
                    "text_label": 英文文本(row),
                    "category_en": category_info.get("en", ""),
                    "light_conditions": row.get("light_conditions", ""),
                    "weather": row.get("weather", ""),
                    "scene": row.get("scene", ""),
                    "source_category": row.get("category", ""),
                }
            )
        except Exception as exc:
            skipped.append(f"{vid}: {exc}")

        if index % 25 == 0 or index == len(rows):
            打印(f"[进度] {index}/{len(rows)}，有效样本 {len(output_rows)}，跳过 {len(skipped)}")

    if skipped:
        skip_path = PROJECT_ROOT / "label_data" / "meta0713_skipped.txt"
        skip_path.write_text("\n".join(skipped), encoding="utf-8")
        打印(f"[警告] 有 {len(skipped)} 条样本跳过，详情：{skip_path}")
    return output_rows


def 写csv(rows: Iterable[Dict[str, str]], output_csv: Path) -> None:
    fieldnames = [
        "video_name",
        "frame_paths",
        "frame_numbers",
        "text_label",
        "category_en",
        "light_conditions",
        "weather",
        "scene",
        "source_category",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    打印(f"[完成] CLIP 数据集已写出：{output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 meta0713.xls 并生成 CLIP/LoRA 数据集")
    parser.add_argument("--original-meta", type=Path, default=DEFAULT_ORIGINAL_META)
    parser.add_argument("--meta-copy", type=Path, default=DEFAULT_META_COPY)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--frame-root", type=Path, default=DEFAULT_FRAME_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--field-report", type=Path, default=DEFAULT_FIELD_REPORT)
    parser.add_argument("--frame-count", type=int, default=5)
    parser.add_argument("--fixed-last-frame", type=int, default=DEFAULT_FIXED_LAST_FRAME, help="固定抽帧序列最后一帧的 1 基帧号；设为 0 表示不固定")
    parser.add_argument("--overwrite-frames", action="store_true", help="重新覆盖已抽取帧")
    args = parser.parse_args()

    复制源表格(args.original_meta, args.meta_copy)
    rows, headers = 读取元数据(args.meta_copy)
    打印(f"[字段] 行数={len(rows)}，字段={headers}")
    for name in headers:
        打印(f"  - {name}: {FIELD_MEANINGS.get(name, '未定义字段')}")
    写字段报告(rows, headers, args.field_report)

    fixed_last_frame = args.fixed_last_frame if args.fixed_last_frame > 0 else None
    dataset_rows = 构建数据集(rows, args.video_root, args.frame_root, args.frame_count, fixed_last_frame, args.overwrite_frames)
    写csv(dataset_rows, args.output_csv)
    打印(f"[统计] 输出样本数={len(dataset_rows)}")


if __name__ == "__main__":
    main()
