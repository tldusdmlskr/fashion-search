"""
R2 labeling/ 전체 JSON 키를 인덱싱해 tasks.json 생성 (스트라이프·스타일당 N건 제한 없음).

S3 list_objects 만 사용 — JSON 본문 다운로드 없음.

실행:
  python annotation_setting/collect_all_tasks.py
  python annotation_setting/collect_all_tasks.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from constants import STYLES  # noqa: E402
from r2_io import R2Config, create_r2_client, get_thread_client, iter_label_keys  # noqa: E402


@dataclass
class LabelIndexHit:
    file_id: int
    style: str
    label_key: str


@dataclass
class StyleListResult:
    style: str
    hits: list[LabelIndexHit]
    elapsed_s: float
    error: str | None = None


def _style_from_key(label_key: str) -> str:
    parts = label_key.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "labeling":
        return parts[1]
    if len(parts) >= 2:
        return parts[-2]
    return ""


def _file_id_from_key(label_key: str) -> int:
    return int(Path(label_key).stem)


def _style_config(base: R2Config, style: str) -> R2Config:
    return R2Config(
        account_id=base.account_id,
        access_key=base.access_key,
        secret_key=base.secret_key,
        bucket=base.bucket,
        image_prefix=base.image_prefix,
        label_prefix=f"labeling/{style.strip('/')}/",
    )


def discover_styles(s3_client, config: R2Config) -> list[str]:
    """labeling/ 하위 공통 prefix(스타일 폴더) 목록."""
    prefix = config.label_prefix or "labeling/"
    if not prefix.endswith("/"):
        prefix += "/"

    styles: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=config.bucket, Prefix=prefix, Delimiter="/"
    ):
        for cp in page.get("CommonPrefixes", []):
            p = cp.get("Prefix", "")
            name = p.rstrip("/").split("/")[-1]
            if name:
                styles.append(name)
    return sorted(styles)


def _list_one_style(style: str, base_config: R2Config) -> StyleListResult:
    t0 = time.perf_counter()
    style_config = _style_config(base_config, style)
    s3_client = get_thread_client(style_config)
    hits: list[LabelIndexHit] = []

    try:
        for label_key in iter_label_keys(s3_client, style_config):
            try:
                file_id = _file_id_from_key(label_key)
            except ValueError:
                continue
            hits.append(
                LabelIndexHit(
                    file_id=file_id,
                    style=style,
                    label_key=label_key,
                )
            )
    except Exception as exc:
        return StyleListResult(
            style=style, hits=hits, elapsed_s=time.perf_counter() - t0, error=str(exc)
        )

    return StyleListResult(style=style, hits=hits, elapsed_s=time.perf_counter() - t0)


def collect_all_indices(
    *,
    styles: list[str] | None = None,
    style_workers: int = 16,
    discover: bool = True,
) -> tuple[list[LabelIndexHit], dict[str, int]]:
    base_config = R2Config.from_defaults()
    s3 = create_r2_client(base_config)

    if styles:
        target_styles = styles
    elif discover:
        print("[DISCOVER] labeling/ 하위 스타일 폴더 탐색…")
        target_styles = discover_styles(s3, base_config)
        if not target_styles:
            target_styles = STYLES
    else:
        target_styles = STYLES

    max_workers = max(1, min(style_workers, len(target_styles)))
    print(
        f"[START] bucket={base_config.bucket}, styles={len(target_styles)}, "
        f"style_workers={max_workers} (full index, no filter)"
    )

    results: dict[str, StyleListResult] = {}
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_list_one_style, style, base_config): style
            for style in target_styles
        }
        for fut in as_completed(futures):
            result = fut.result()
            results[result.style] = result
            if result.error:
                print(
                    f"[ERROR] {result.style}: {result.error} "
                    f"(partial={len(result.hits)}, {result.elapsed_s:.1f}s)"
                )
            else:
                print(f"[OK] {result.style}: {len(result.hits)} json ({result.elapsed_s:.1f}s)")

    hits: list[LabelIndexHit] = []
    count_per_style: dict[str, int] = {}
    for style in target_styles:
        r = results.get(style)
        if r is None:
            continue
        hits.extend(r.hits)
        count_per_style[style] = len(r.hits)

    print(f"[DONE] total={len(hits)}, elapsed={time.perf_counter() - t0:.1f}s")
    return hits, count_per_style


def build_tasks_payload(
    hits: list[LabelIndexHit],
    *,
    styles: list[str],
    count_per_style: dict[str, int],
) -> dict[str, Any]:
    by_style: dict[str, list[int]] = {s: [] for s in styles}
    for hit in hits:
        by_style.setdefault(hit.style, []).append(hit.file_id)

    return {
        "source": "r2_labeling_full_index",
        "filter": None,
        "per_style_limit": None,
        "target_total": len(hits),
        "actual_total": len(hits),
        "indices": [h.file_id for h in hits],
        "by_style": by_style,
        "records": [asdict(h) for h in hits],
        "json_count_per_style": count_per_style,
        "styles": styles,
    }


def save_tasks_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R2 labeling/ 전체 JSON 키 인덱스 → tasks.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR / "tasks.json",
    )
    parser.add_argument(
        "--style",
        action="append",
        dest="styles",
        help="특정 스타일만 (미지정 시 버킷에서 자동 탐색)",
    )
    parser.add_argument(
        "--no-discover",
        action="store_true",
        help="자동 탐색 끄고 constants.STYLES 24개만 사용",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    hits, count_per_style = collect_all_indices(
        styles=args.styles,
        style_workers=args.workers,
        discover=not args.no_discover,
    )

    styles_used = sorted(count_per_style.keys())
    payload = build_tasks_payload(hits, styles=styles_used, count_per_style=count_per_style)
    save_tasks_json(payload, args.output)

    print(f"\n완료: {payload['actual_total']}건 → {args.output}")


if __name__ == "__main__":
    main()
