# 마스크 데이터를 R2로 업로드하는 스크립트

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
VLM_DIR = REPO_ROOT / "VLM_captioning"
if str(VLM_DIR) not in sys.path:
    sys.path.insert(0, str(VLM_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(VLM_DIR / ".env")
except ImportError:
    pass

from botocore.exceptions import ClientError

from vlm_common import R2Config, create_r2_client


_THREAD_LOCAL = threading.local()
_MANIFEST_LOCK = threading.Lock()


def _load_r2_env_fallback() -> None:
    """VLM_captioning/.env에 R2 값이 없을 때 로컬 설정 파일에서 보충."""
    if os.environ.get("R2_ACCOUNT_ID"):
        return

    fallback = REPO_ROOT / "label_studio_setting" / "json_convert.py"
    if not fallback.exists():
        return

    text = fallback.read_text(encoding="utf-8")
    mapping = {
        "R2_ACCOUNT_ID": r'R2_ACCOUNT_ID\s*=\s*"([^"]+)"',
        "R2_ACCESS_KEY_ID": r'R2_ACCESS_KEY\s*=\s*"([^"]+)"',
        "R2_SECRET_ACCESS_KEY": r'R2_SECRET_KEY\s*=\s*"([^"]+)"',
    }
    for env_name, pattern in mapping.items():
        match = re.search(pattern, text)
        if match:
            os.environ[env_name] = match.group(1)


def _load_uploaded_manifest(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    return {
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _append_manifest(manifest_path: Path, rel_name: str) -> None:
    with _MANIFEST_LOCK:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(rel_name + "\n")


def _thread_client(config: R2Config):
    if not hasattr(_THREAD_LOCAL, "s3"):
        _THREAD_LOCAL.s3 = create_r2_client(config)
    return _THREAD_LOCAL.s3


def _list_existing_remote_names(s3_client, *, bucket: str, prefix: str) -> set[str]:
    existing: set[str] = set()
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            existing.add(Path(key).name)
    return existing


def _upload_one(
    file_path: Path,
    *,
    config: R2Config,
    remote_prefix: str,
    retries: int,
) -> tuple[str, str]:
    """
    반환: (상태, 파일명)
    상태: uploaded | failed
    """
    s3 = _thread_client(config)
    key = f"{remote_prefix}{file_path.name}"

    for attempt in range(retries + 1):
        try:
            s3.upload_file(str(file_path), config.bucket, key)
            return "uploaded", file_path.name
        except ClientError:
            if attempt >= retries:
                return "failed", file_path.name
            time.sleep(min(2**attempt, 10))
        except Exception:
            if attempt >= retries:
                return "failed", file_path.name
            time.sleep(min(2**attempt, 10))

    return "failed", file_path.name


def upload_masking_data(
    *,
    local_dir: Path,
    bucket: str,
    remote_prefix: str,
    workers: int,
    retries: int,
    manifest_path: Path,
    skip_remote_scan: bool,
) -> None:
    _load_r2_env_fallback()
    config = R2Config.from_env()
    config.bucket = bucket

    if remote_prefix and not remote_prefix.endswith("/"):
        remote_prefix += "/"

    local_files = sorted(local_dir.glob("*.jpg"))
    if not local_files:
        raise FileNotFoundError(f"JPG 파일이 없습니다: {local_dir}")

    remote_existing: set[str] = set()
    if not skip_remote_scan:
        base_client = create_r2_client(config)
        remote_existing = _list_existing_remote_names(
            base_client,
            bucket=config.bucket,
            prefix=remote_prefix,
        )
    done_from_manifest = _load_uploaded_manifest(manifest_path)
    done_names = remote_existing | done_from_manifest

    queue = [p for p in local_files if p.name not in done_names]
    total = len(local_files)
    skipped = total - len(queue)

    print(
        f"[START] total={total}, skipped_existing={skipped}, "
        f"to_upload={len(queue)}, workers={workers}"
    )
    print(f"[TARGET] bucket={config.bucket}, prefix={remote_prefix}")
    print(f"[MODE] skip_remote_scan={skip_remote_scan}")
    print(f"[MANIFEST] {manifest_path}")

    uploaded = 0
    failed = 0
    max_workers = max(1, workers)
    max_in_flight = max_workers * 6

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight: dict[Future, Path] = {}
        next_index = 0

        while next_index < len(queue) and len(in_flight) < max_in_flight:
            p = queue[next_index]
            next_index += 1
            fut = executor.submit(
                _upload_one,
                p,
                config=config,
                remote_prefix=remote_prefix,
                retries=retries,
            )
            in_flight[fut] = p

        processed = 0
        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                p = in_flight.pop(fut)
                status, name = fut.result()
                processed += 1
                if status == "uploaded":
                    uploaded += 1
                    _append_manifest(manifest_path, name)
                else:
                    failed += 1
                    print(f"[FAIL] {name}")

                if processed % 500 == 0 or processed == len(queue):
                    print(
                        f"[PROGRESS] uploaded={uploaded}, failed={failed}, "
                        f"done={processed}/{len(queue)}"
                    )

            while next_index < len(queue) and len(in_flight) < max_in_flight:
                p = queue[next_index]
                next_index += 1
                fut = executor.submit(
                    _upload_one,
                    p,
                    config=config,
                    remote_prefix=remote_prefix,
                    retries=retries,
                )
                in_flight[fut] = p

    print(
        f"[RESULT] total={total}, skipped_existing={skipped}, "
        f"uploaded={uploaded}, failed={failed}"
    )
    if failed > 0:
        print("[WARNING] 실패 파일이 있어 재실행 권장 (manifest 기반으로 나머지만 재시도됨).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="masking_data JPG를 Cloudflare R2로 안정 업로드"
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "masking_data",
        help="업로드할 로컬 폴더",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="project2",
        help="R2 버킷명",
    )
    parser.add_argument(
        "--remote-prefix",
        type=str,
        default="masking_data/",
        help="R2 목적지 prefix",
    )
    parser.add_argument("--workers", type=int, default=16, help="병렬 업로드 워커 수")
    parser.add_argument("--retries", type=int, default=4, help="업로드 재시도 횟수")
    parser.add_argument(
        "--skip-remote-scan",
        action="store_true",
        help="원격 prefix 전체 목록 스캔 생략 (manifest 기반 재개)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(__file__).resolve().parent / "upload_manifest.txt",
        help="업로드 완료 파일 manifest 경로",
    )
    args = parser.parse_args()

    upload_masking_data(
        local_dir=args.local_dir,
        bucket=args.bucket,
        remote_prefix=args.remote_prefix,
        workers=args.workers,
        retries=args.retries,
        manifest_path=args.manifest_path,
        skip_remote_scan=args.skip_remote_scan,
    )


if __name__ == "__main__":
    main()
