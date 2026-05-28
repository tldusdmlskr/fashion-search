"""R2 labeling JSON 대비 masking_data 누락 검증."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VLM_DIR = REPO_ROOT / "VLM_captioning"
if str(VLM_DIR) not in sys.path:
    sys.path.insert(0, str(VLM_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(VLM_DIR / ".env")
except ImportError:
    pass

from vlm_common import (  # noqa: E402
    R2Config,
    create_r2_client,
    iter_label_keys,
    load_json_from_r2,
)


def _load_r2_env_fallback() -> None:
    """VLM_captioning/.env에 R2 값이 없을 때 로컬 설정 파일에서 보충."""
    if os.environ.get("R2_ACCOUNT_ID"):
        return

    fallback = REPO_ROOT / "label_studio_setting" / "json_convert.py"
    if not fallback.exists():
        return

    text = fallback.read_text(encoding="utf-8")
    mapping = {
        "R2_ACCOUNT_ID": r'R2_ACCOUNT_ID\s*=\s*"([^"]+)"',
        "R2_ACCESS_KEY_ID": r'R2_ACCESS_KEY\s*=\s*"([^"]+)"',
        "R2_SECRET_ACCESS_KEY": r'R2_SECRET_KEY\s*=\s*"([^"]+)"',
    }
    for env_name, pattern in mapping.items():
        match = re.search(pattern, text)
        if match:
            os.environ[env_name] = match.group(1)

CATEGORY_MAP = {
    "아우터": "outerwear",
    "원피스": "dress",
    "상의": "top",
    "하의": "bottom",
}
MAX_POLYGON_POINTS = 64
_THREAD_LOCAL = threading.local()


def _extract_polygon_points(polygon_item: dict[str, Any]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for idx in range(1, MAX_POLYGON_POINTS + 1):
        x_key = f"X좌표{idx}"
        y_key = f"Y좌표{idx}"
        if x_key not in polygon_item or y_key not in polygon_item:
            break
        points.append((int(polygon_item[x_key]), int(polygon_item[y_key])))
    return points


def _image_id_from_label(raw: dict[str, Any], label_key: str) -> str:
    image_info = raw.get("이미지 정보", {})
    dataset_info = raw.get("데이터셋 정보", {})

    image_id = image_info.get("이미지 식별자")
    if image_id not in (None, ""):
        return str(image_id)

    file_number = dataset_info.get("파일 번호")
    if file_number not in (None, ""):
        return str(file_number)

    return Path(label_key).stem


def expected_mask_names(raw: dict[str, Any], label_key: str) -> list[str]:
    """폴리곤이 유효한 카테고리별 기대 파일명."""
    polygon_root = (
        raw.get("데이터셋 정보", {})
        .get("데이터셋 상세설명", {})
        .get("폴리곤좌표", {})
    )
    image_id = _image_id_from_label(raw, label_key)
    expected: list[str] = []

    for category_ko, category_en in CATEGORY_MAP.items():
        polygon_items = polygon_root.get(category_ko, [])
        if not isinstance(polygon_items, list):
            continue

        has_valid = False
        for polygon_item in polygon_items:
            if not polygon_item:
                continue
            if len(_extract_polygon_points(polygon_item)) >= 3:
                has_valid = True
                break

        if has_valid:
            expected.append(f"{image_id}_{category_en}.jpg")

    return expected


def _get_thread_client(config: R2Config):
    if not hasattr(_THREAD_LOCAL, "s3_client"):
        _THREAD_LOCAL.s3_client = create_r2_client(config)
    return _THREAD_LOCAL.s3_client


def _check_label(
    label_key: str,
    *,
    config: R2Config,
    local_files: set[str],
) -> dict[str, Any]:
    s3_client = _get_thread_client(config)
    try:
        raw = load_json_from_r2(s3_client, config, label_key)
    except Exception as exc:
        return {
            "label_key": label_key,
            "status": "json_error",
            "error": str(exc),
            "missing": [],
            "expected": [],
        }

    expected = expected_mask_names(raw, label_key)
    missing = [name for name in expected if name not in local_files]

    if missing:
        status = "missing_masks"
    elif expected:
        status = "ok"
    else:
        status = "no_polygon"

    return {
        "label_key": label_key,
        "status": status,
        "error": None,
        "missing": missing,
        "expected": expected,
    }


def validate(
    *,
    masking_dir: Path,
    output_dir: Path,
    workers: int = 16,
    limit: int | None = None,
) -> dict[str, Any]:
    _load_r2_env_fallback()
    config = R2Config.from_env()
    config.bucket = "project2"
    config.label_prefix = "labeling/"

    s3_client = create_r2_client(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_files = {p.name for p in masking_dir.glob("*.jpg")}
    print(f"[INFO] local mask files: {len(local_files)}")

    label_keys: list[str] = []
    for label_key in iter_label_keys(s3_client, config):
        label_keys.append(label_key)
        if limit is not None and len(label_keys) >= limit:
            break

    total_labels = len(label_keys)
    print(f"[INFO] R2 label json count: {total_labels}")

    stats = {
        "total_labels": total_labels,
        "ok": 0,
        "no_polygon": 0,
        "missing_masks": 0,
        "json_error": 0,
        "expected_masks": 0,
        "missing_mask_files": 0,
        "local_orphan_files": 0,
    }

    missing_rows: list[dict[str, str]] = []
    error_rows: list[dict[str, str]] = []
    expected_names: set[str] = set()

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                _check_label,
                label_key,
                config=config,
                local_files=local_files,
            )
            for label_key in label_keys
        ]

        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            status = result["status"]
            stats[status] += 1
            stats["expected_masks"] += len(result["expected"])
            stats["missing_mask_files"] += len(result["missing"])

            for name in result["expected"]:
                expected_names.add(name)

            if result["missing"]:
                stats["missing_masks"] += 1
                for name in result["missing"]:
                    missing_rows.append(
                        {
                            "label_key": result["label_key"],
                            "missing_file": name,
                        }
                    )

            if status == "json_error":
                error_rows.append(
                    {
                        "label_key": result["label_key"],
                        "error": result["error"] or "",
                    }
                )

            if done % 5000 == 0 or done == total_labels:
                print(f"[PROGRESS] {done}/{total_labels}")

    orphan_files = sorted(local_files - expected_names)
    stats["local_orphan_files"] = len(orphan_files)

    missing_path = output_dir / "missing_masks.jsonl"
    with missing_path.open("w", encoding="utf-8") as f:
        for row in missing_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    error_path = output_dir / "json_errors.jsonl"
    with error_path.open("w", encoding="utf-8") as f:
        for row in error_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    orphan_path = output_dir / "orphan_local_files.txt"
    orphan_path.write_text("\n".join(orphan_files), encoding="utf-8")

    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[RESULT]")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")
    print(f"missing detail: {missing_path}")
    print(f"json errors: {error_path}")
    print(f"orphan files: {orphan_path}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="masking_data 누락 검증")
    parser.add_argument(
        "--masking-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "masking_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "validation_report",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    validate(
        masking_dir=args.masking_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
