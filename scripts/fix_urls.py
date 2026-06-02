"""
fix_urls.py — output/ 내 tasks*.json 파일의 이미지 URL 한글 → 영문 일괄 수정
실행: python scripts/fix_urls.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.url_utils import fix_url

OUTPUT_DIR = Path(__file__).parent.parent / "output"

targets = list(OUTPUT_DIR.glob("tasks*.json"))
if not targets:
    print("output/ 에 tasks*.json 파일이 없습니다.")
else:
    for p in targets:
        with open(p, encoding="utf-8") as f:
            tasks = json.load(f)

        changed = 0
        for task in tasks:
            old_url = task["data"]["image"]
            new_url = fix_url(old_url)
            if old_url != new_url:
                task["data"]["image"] = new_url
                changed += 1

        with open(p, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        print(f"{p.name}: {changed}/{len(tasks)}개 URL 수정")
