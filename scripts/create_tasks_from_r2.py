"""
R2 라벨 JSON(로컬 없음) → Label Studio task JSON 생성.

1) annotation_setting/tasks.json 의 records 사용 (collect_stripe_tasks.py 결과)
2) 또는 --file-ids 로 ID 목록 직접 지정 (style은 --default-style 필요)

실행 예:
  python scripts/create_tasks_from_r2.py
  python scripts/create_tasks_from_r2.py --clothing-type 하의
  python scripts/create_tasks_from_r2.py --tasks-index annotation_setting/tasks.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "annotation_setting"))

from r2_io import R2Config, create_r2_client, load_json_from_r2  # noqa: E402
from src.label_parser import rows_from_label_json  # noqa: E402
from src.task_builder import make_tasks_from_df  # noqa: E402


def _load_records(tasks_index_path: Path) -> list[dict]:
    payload = json.loads(tasks_index_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if records:
        return records
    # indices + by_style 만 있는 경우
    by_style = payload.get("by_style", {})
    out: list[dict] = []
    for style, ids in by_style.items():
        for fid in ids:
            out.append({"file_id": fid, "style": style, "label_key": ""})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="R2 라벨 → Label Studio tasks.json")
    parser.add_argument(
        "--tasks-index",
        type=Path,
        default=ROOT / "annotation_setting" / "tasks.json",
        help="collect_stripe_tasks.py 출력 JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "tasks.json",
        help="Label Studio import용 task JSON",
    )
    parser.add_argument(
        "--clothing-type",
        default=None,
        help="특정 의류 타입만 task 생성 (예: 하의)",
    )
    parser.add_argument(
        "--default-style",
        default="",
        help="records에 style이 없을 때 사용",
    )
    parser.add_argument("--limit", type=int, default=None, help="테스트용 최대 건수")
    args = parser.parse_args()

    if not args.tasks_index.is_file():
        resolved = args.tasks_index.resolve()
        raise SystemExit(
            f"인덱스 없음: {args.tasks_index}\n"
            f"  (확인 경로: {resolved})\n"
            "먼저 샘플링을 실행하세요. 예:\n"
            "  python annotation_setting/collect_bottom_tasks.py "
            "--output annotation_setting/tasks2.json"
        )

    records = _load_records(args.tasks_index)
    if args.limit:
        records = records[: args.limit]

    cfg = R2Config.from_defaults()
    s3 = create_r2_client(cfg)

    rows: list[dict] = []
    missing = 0
    for i, rec in enumerate(records, 1):
        file_id = int(rec["file_id"])
        style = rec.get("style") or args.default_style
        label_key = rec.get("label_key")
        if label_key:
            raw = load_json_from_r2(s3, cfg, label_key)
        else:
            # style 폴더 규칙: labeling/{style}/{file_id}.json
            key = f"{cfg.label_prefix}{style}/{file_id}.json"
            try:
                raw = load_json_from_r2(s3, cfg, key)
            except Exception:
                missing += 1
                continue

        for row in rows_from_label_json(raw, style=style):
            if args.clothing_type and row["clothing_type"] != args.clothing_type:
                continue
            rows.append(row)

        if i % 50 == 0:
            print(f"  R2 로드 {i}/{len(records)} …")

    if not rows:
        raise SystemExit("생성할 row가 없습니다. R2 키·style·clothing-type 필터를 확인하세요.")

    df = pd.DataFrame(rows)
    tasks = make_tasks_from_df(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\n→ {args.out} 저장 ({len(tasks)} tasks, rows={len(df)}, R2 miss={missing})")
    if tasks:
        print(json.dumps(tasks[0], ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    main()
