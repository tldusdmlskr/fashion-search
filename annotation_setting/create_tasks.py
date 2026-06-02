"""
로컬 data/labels/ 가 있을 때만 사용.

클라우드(R2)만 쓰는 경우:
  python annotation_setting/collect_stripe_tasks.py
  python scripts/create_tasks_from_r2.py
  python scripts/upload_tasks_to_ls.py

로컬 라벨 + 다양성 샘플링:
  python scripts/create_tasks.py
"""
raise SystemExit(__doc__)
