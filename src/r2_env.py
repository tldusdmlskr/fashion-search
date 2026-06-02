"""annotation_setting/r2_credentials.py → os.environ (로컬 .env 없을 때 R2용)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def apply_annotation_r2_credentials() -> bool:
    """
    .env에 R2 값이 없으면 annotation_setting/r2_credentials.py 를 읽어 채운다.
    Returns True if credentials module was loaded.
    """
    if os.environ.get("R2_ACCESS_KEY_ID") and os.environ.get("R2_SECRET_ACCESS_KEY"):
        return False

    cred_path = Path(__file__).resolve().parent.parent / "annotation_setting" / "r2_credentials.py"
    if not cred_path.is_file():
        return False

    spec = importlib.util.spec_from_file_location("annotation_r2_credentials", cred_path)
    if spec is None or spec.loader is None:
        return False
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    account_id = getattr(mod, "R2_ACCOUNT_ID", "") or ""
    access_key = getattr(mod, "R2_ACCESS_KEY", "") or getattr(mod, "R2_ACCESS_KEY_ID", "")
    secret_key = getattr(mod, "R2_SECRET_KEY", "") or getattr(mod, "R2_SECRET_ACCESS_KEY", "")
    bucket = getattr(mod, "R2_BUCKET", "project2")

    if account_id and not os.environ.get("R2_ENDPOINT_URL"):
        os.environ["R2_ENDPOINT_URL"] = f"https://{account_id}.r2.cloudflarestorage.com"
    if access_key:
        os.environ.setdefault("R2_ACCESS_KEY_ID", access_key)
    if secret_key:
        os.environ.setdefault("R2_SECRET_ACCESS_KEY", secret_key)
    os.environ.setdefault("R2_BUCKET", bucket)

    public_url = getattr(mod, "R2_PUBLIC_URL", "") or ""
    if public_url:
        os.environ.setdefault("R2_PUBLIC_URL", public_url.rstrip("/"))

    return True
