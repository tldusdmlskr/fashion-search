"""tasks.json(bottom_sampling)에서 sample_group=drop_waist 목록 출력."""
from __future__ import annotations

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks-index",
        type=Path,
        default=THIS_DIR / "tasks.json",
    )
    args = parser.parse_args()
    tasks_path = args.tasks_index

    if not tasks_path.is_file():
        print(f"없음: {tasks_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(tasks_path.read_text(encoding="utf-8"))
    source = data.get("source", "")
    records = data.get("records", [])

    drop = [r for r in records if r.get("sample_group") == "drop_waist"]

    print(f"source: {source}")
    print(f"actual_total: {data.get('actual_total')}")
    print(f"drop_waist selected: {len(drop)}")
    print()
    if not drop:
        print("(드롭웨이스트로 뽑힌 샘플 없음)")
        stats = data.get("per_style_stats") or {}
        if stats:
            print("\n[스타일별 pool_drop_waist (후보 수)]")
            for style in sorted(stats):
                p = stats[style].get("pool_drop_waist", 0)
                s = stats[style].get("selected_drop_waist", 0)
                if p or s:
                    print(f"  {style}: pool={p}, selected={s}")
        sys.exit(0)

    print("style\tfile_id\tlabel_key")
    for r in sorted(drop, key=lambda x: (x["style"], x["file_id"])):
        print(f"{r['style']}\t{r['file_id']}\t{r.get('label_key', '')}")


if __name__ == "__main__":
    main()
