"""
update_ls_tasks.py — Label Studio 기존 task data를 tasks_final.json 기준으로 업데이트
실행: python scripts/update_ls_tasks.py [tasks_json_path]
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ls_client import get_project, get_all_tasks, patch_task

TASKS_PATH = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).parent.parent / "output" / "tasks_final.json"
)

# 인증 테스트
resp = get_project()
if resp.status_code != 200:
    print(f"인증 실패 ({resp.status_code}): {resp.text[:200]}")
    raise SystemExit(1)
print(f"프로젝트: {resp.json().get('title')}")

# LS task 매핑 로드
print("\nLS task 목록 가져오는 중...")
ls_map = get_all_tasks()   # {file_id: ls_task_id}
print(f"총 {len(ls_map)}개 task 매핑 완료")

# 업데이트
with open(TASKS_PATH, encoding="utf-8") as f:
    new_tasks = json.load(f)

updated, skipped = 0, 0
for task in new_tasks:
    fid   = task["data"]["file_id"]
    ls_id = ls_map.get(fid)
    if ls_id is None:
        skipped += 1
        continue

    resp = patch_task(ls_id, task["data"])
    if resp.status_code in (200, 201):
        updated += 1
        if updated % 100 == 0:
            print(f"  {updated}개 업데이트 완료...")
    else:
        print(f"  FAIL task {ls_id} (file_id={fid}): {resp.status_code}")

print(f"\n완료: {updated}개 업데이트, {skipped}개 스킵(LS에 없음)")
