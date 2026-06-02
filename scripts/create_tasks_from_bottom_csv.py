"""Create Label Studio tasks from bottom_low_confidence_250.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "annotation_setting" / "bottom_low_confidence_250.csv"
OUT_PATH = ROOT / "output" / "tasks_bottom.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--image-field",
        choices=["image_url", "masking_url"],
        default="image_url",
        help="Label Studio data.image에 넣을 URL 필드",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")

    tasks = []
    for _, row in df.iterrows():
        file_id = int(row["image_id"])
        item_info = "\n".join(
            [
                "타입: 하의",
                f"스타일: {row.get('folder', '')}",
                f"length_pred: {row.get('length_pred', '')} ({row.get('length_conf', '')})",
                f"waist_pred: {row.get('waist_pred', '')} ({row.get('waist_conf', '')})",
                f"total_conf: {row.get('total_conf', '')}",
            ]
        )

        tasks.append(
            {
                "data": {
                    "image": row.get(args.image_field) or row.get("image_url") or row.get("masking_url"),
                    "source_image": row.get("image_url", ""),
                    "masking_url": row.get("masking_url", ""),
                    "item_info": item_info,
                    "file_id": file_id,
                    "folder": row.get("folder", ""),
                    "length_pred": row.get("length_pred", ""),
                    "waist_pred": row.get("waist_pred", ""),
                    "length_conf": str(row.get("length_conf", "")),
                    "waist_conf": str(row.get("waist_conf", "")),
                    "total_conf": str(row.get("total_conf", "")),
                },
                "predictions": [],
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"saved: {args.out} ({len(tasks)} tasks)")


if __name__ == "__main__":
    main()

