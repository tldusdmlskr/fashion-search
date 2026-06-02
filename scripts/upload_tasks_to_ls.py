"""
output/tasks.json → Label Studio 프로젝트에 일괄 import.

실행:
  python scripts/upload_tasks_to_ls.py
  python scripts/upload_tasks_to_ls.py --tasks output/tasks.json
  python scripts/test_auth.py   # 인증 먼저 확인
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ls_client import import_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="tasks.json → Label Studio import")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "output" / "tasks.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="업로드 없이 건수만 출력",
    )
    args = parser.parse_args()

    if not args.tasks.is_file():
        raise SystemExit(f"파일 없음: {args.tasks}\n먼저 create_tasks_from_r2.py 또는 create_tasks.py 실행")

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    print(f"tasks: {len(tasks)}건")
    if args.dry_run:
        return

    result = import_tasks(tasks)
    print(f"import 완료: {result}")


if __name__ == "__main__":
    main()
