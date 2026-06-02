import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from src.r2_env import apply_annotation_r2_credentials  # noqa: E402

apply_annotation_r2_credentials()

# R2 (create_tasks / export_metadata / R2 스크립트)
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "project2")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "")
R2_BUCKET_PREFIX = "image"

# Gemini (캡션 스크립트만 필요)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Label Studio (upload_tasks_to_ls.py 등)
LS_URL = os.environ.get("LS_URL", "")
LS_API_TOKEN = os.environ.get("LS_API_TOKEN", "")
LS_PROJECT_ID = int(os.environ.get("LS_PROJECT_ID", "0") or 0)

# Turso / SQLite (optional)
DB_URL = os.environ.get("DB_URL")
DB_ACCESS_TOKEN = os.environ.get("DB_ACCESS_TOKEN")

# 도메인 상수
CLOTHING_TYPES = ['상의', '하의', '아우터', '원피스']
