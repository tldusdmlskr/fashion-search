"""
query_ls_tasks.py — 특정 Label Studio task ID → file_id 조회
실행: python scripts/query_ls_tasks.py [task_id ...]

예: python scripts/query_ls_tasks.py 24310 24316 24375
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ls_client import get_task

TARGET_IDS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [24310, 24316, 24375, 24486, 24555]

print("Label Studio task ID → file_id 매핑:")
print("-" * 50)

for task_id in TARGET_IDS:
    resp = get_task(task_id)
    if resp.status_code == 200:
        data      = resp.json()
        file_id   = data.get("data", {}).get("file_id", "N/A")
        item_info = data.get("data", {}).get("item_info", "")
        image     = data.get("data", {}).get("image", "")
        print(f"  LS task {task_id} → file_id={file_id}")
        if item_info:
            print(f"    item_info: '{item_info[:30]}...'")
        else:
            print(f"    item_info: (비어있음)")
        print(f"    image: {image.split('/')[-1]}")
    else:
        print(f"  LS task {task_id} → 조회 실패 ({resp.status_code})")
    print()
