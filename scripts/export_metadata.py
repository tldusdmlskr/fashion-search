"""
export_metadata.py — R2 버킷의 labeling/ 하위 JSON을 data/labels/ 로 다운로드
실행: python scripts/export_metadata.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.r2_client import get_r2_client
from src.config import R2_BUCKET

LABEL_PREFIX = "labeling/"
LOCAL_DIR    = Path(__file__).parent.parent / "data" / "labels"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

s3 = get_r2_client()

paginator = s3.get_paginator("list_objects_v2")
downloaded = 0
for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=LABEL_PREFIX):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".json"):
            continue
        local_path = LOCAL_DIR / Path(key).name
        s3.download_file(R2_BUCKET, key, str(local_path))
        downloaded += 1
        if downloaded % 1000 == 0:
            print(f"  {downloaded}개 다운로드 중...")

print(f"완료: {downloaded}개 JSON → {LOCAL_DIR}")
