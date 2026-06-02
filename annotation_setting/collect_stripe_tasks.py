"""
R2 labeling JSON을 스타일별로 탐색해 프린트=스트라이프 이미지 인덱스를 수집하고 tasks.json에 저장.

스타일당 기본 40건 × 24스타일 = 960건 목표.
R2 접근 키는 annotation_setting/r2_credentials.py 에 하드코딩.

실행 예:
  python annotation_setting/collect_stripe_tasks.py
  python annotation_setting/collect_stripe_tasks.py --workers 12
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

from r2_io import (  # noqa: E402
    R2Config,
    get_thread_client,
    iter_label_keys,
    load_json_from_r2,
)

from constants import (  # noqa: E402
    CLOTHING_TYPES,
    DEFAULT_PER_STYLE,
    PRINT_ATTR,
    PRINT_VALUE,
    STYLES,
    TARGET_TOTAL,
)


@dataclass
class StripeHit:
    file_id: int
    style: str
    label_key: str


@dataclass
class StyleCollectResult:
    style: str
    hits: list[StripeHit]
    scanned: int
    elapsed_s: float
    error: str | None = None


def _file_id_from_label(raw: dict[str, Any], label_key: str) -> int:
    dataset_info = raw.get("데이터셋 정보", {})
    file_number = dataset_info.get("파일 번호")
    if file_number not in (None, ""):
        return int(file_number)

    image_info = raw.get("이미지 정보", {})
    image_id = image_info.get("이미지 식별자")
    if image_id not in (None, ""):
        return int(image_id)

    return int(Path(label_key).stem)


def _has_stripe_print(raw: dict[str, Any]) -> bool:
    labeling = (
        raw.get("데이터셋 정보", {})
        .get("데이터셋 상세설명", {})
        .get("라벨링", {})
    )
    if not isinstance(labeling, dict):
        return False

    for clothing_type in CLOTHING_TYPES:
        items = labeling.get(clothing_type, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or not item:
                continue
            prints = item.get(PRINT_ATTR, [])
            if isinstance(prints, str):
                prints = [prints]
            if PRINT_VALUE in prints:
                return True
    return False


def _style_config(base: R2Config, style: str) -> R2Config:
    return R2Config(
        account_id=base.account_id,
        access_key=base.access_key,
        secret_key=base.secret_key,
        bucket=base.bucket,
        image_prefix=base.image_prefix,
        label_prefix=f"labeling/{style.strip('/')}/",
    )


def _try_stripe_hit(
    s3_client,
    style_config: R2Config,
    style: str,
    label_key: str,
) -> StripeHit | None:
    try:
        raw = load_json_from_r2(s3_client, style_config, label_key)
    except Exception as exc:
        print(f"[SKIP] {label_key}: {exc}")
        return None

    if not _has_stripe_print(raw):
        return None

    return StripeHit(
        file_id=_file_id_from_label(raw, label_key),
        style=style,
        label_key=label_key,
    )


def _collect_one_style(
    style: str,
    base_config: R2Config,
    *,
    per_style: int,
    max_labels_per_style: int | None,
    json_workers: int,
) -> StyleCollectResult:
    t0 = time.perf_counter()
    style_config = _style_config(base_config, style)
    s3_client = get_thread_client(style_config)

    style_hits: list[StripeHit] = []
    scanned = 0
    batch_size = max(8, json_workers * 4)

    try:
        pending_keys: list[str] = []

        def flush_batch(keys: list[str]) -> None:
            nonlocal style_hits
            if not keys or len(style_hits) >= per_style:
                return

            if json_workers <= 1 or len(keys) == 1:
                for label_key in keys:
                    if len(style_hits) >= per_style:
                        return
                    hit = _try_stripe_hit(s3_client, style_config, style, label_key)
                    if hit is not None:
                        style_hits.append(hit)
                return

            with ThreadPoolExecutor(max_workers=json_workers) as pool:
                futures = [
                    pool.submit(_try_stripe_hit, s3_client, style_config, style, key)
                    for key in keys
                ]
                for fut in as_completed(futures):
                    if len(style_hits) >= per_style:
                        break
                    hit = fut.result()
                    if hit is not None:
                        style_hits.append(hit)

        for label_key in iter_label_keys(s3_client, style_config):
            scanned += 1
            if max_labels_per_style is not None and scanned > max_labels_per_style:
                break
            if len(style_hits) >= per_style:
                break

            pending_keys.append(label_key)
            if len(pending_keys) >= batch_size:
                flush_batch(pending_keys)
                pending_keys = []

        if pending_keys and len(style_hits) < per_style:
            flush_batch(pending_keys)

    except Exception as exc:
        return StyleCollectResult(
            style=style,
            hits=style_hits,
            scanned=scanned,
            elapsed_s=time.perf_counter() - t0,
            error=str(exc),
        )

    return StyleCollectResult(
        style=style,
        hits=style_hits[:per_style],
        scanned=scanned,
        elapsed_s=time.perf_counter() - t0,
    )


def collect_stripe_indices(
    *,
    per_style: int = DEFAULT_PER_STYLE,
    styles: list[str] | None = None,
    max_labels_per_style: int | None = None,
    style_workers: int = 12,
    json_workers: int = 4,
) -> tuple[list[StripeHit], dict[str, int]]:
    base_config = R2Config.from_defaults()
    base_config.bucket = "project2"

    target_styles = styles or STYLES
    max_style_workers = max(1, min(style_workers, len(target_styles)))
    max_json_workers = max(1, json_workers)

    print(
        f"[START] bucket={base_config.bucket}, styles={len(target_styles)}, "
        f"per_style={per_style}, target_total={len(target_styles) * per_style}, "
        f"style_workers={max_style_workers}, json_workers={max_json_workers}"
    )

    results_by_style: dict[str, StyleCollectResult] = {}
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_style_workers) as executor:
        futures = {
            executor.submit(
                _collect_one_style,
                style,
                base_config,
                per_style=per_style,
                max_labels_per_style=max_labels_per_style,
                json_workers=max_json_workers,
            ): style
            for style in target_styles
        }
        for fut in as_completed(futures):
            result = fut.result()
            results_by_style[result.style] = result

            if result.error:
                print(
                    f"[ERROR] {result.style}: {result.error} "
                    f"(partial stripe={len(result.hits)}, scanned={result.scanned}, "
                    f"{result.elapsed_s:.1f}s)"
                )
                continue

            status = "OK" if len(result.hits) >= per_style else "SHORT"
            print(
                f"[{status}] {result.style}: stripe={len(result.hits)}/{per_style} "
                f"(scanned {result.scanned} json, {result.elapsed_s:.1f}s)"
            )

    hits: list[StripeHit] = []
    scanned_per_style: dict[str, int] = {}
    for style in target_styles:
        result = results_by_style.get(style)
        if result is None:
            continue
        hits.extend(result.hits)
        scanned_per_style[style] = result.scanned

    print(f"[DONE] total_elapsed={time.perf_counter() - t0:.1f}s")
    return hits, scanned_per_style


def build_tasks_payload(
    hits: list[StripeHit],
    *,
    per_style: int,
    styles: list[str],
    scanned_per_style: dict[str, int],
) -> dict[str, Any]:
    by_style: dict[str, list[int]] = {style: [] for style in styles}
    for hit in hits:
        by_style.setdefault(hit.style, []).append(hit.file_id)

    indices = [hit.file_id for hit in hits]
    records = [asdict(hit) for hit in hits]

    short_styles = [
        style
        for style in styles
        if len(by_style.get(style, [])) < per_style
    ]

    return {
        "filter": {PRINT_ATTR: PRINT_VALUE},
        "per_style_limit": per_style,
        "target_total": len(styles) * per_style,
        "actual_total": len(indices),
        "indices": indices,
        "by_style": by_style,
        "records": records,
        "scanned_json_per_style": scanned_per_style,
        "short_styles": short_styles,
    }


def save_tasks_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R2 labeling JSON에서 프린트=스트라이프 이미지 인덱스 수집 (스타일당 N건)"
    )
    parser.add_argument(
        "--per-style",
        type=int,
        default=DEFAULT_PER_STYLE,
        help=f"스타일당 수집 건수 (기본 {DEFAULT_PER_STYLE}, 24스타일 시 총 {TARGET_TOTAL}건)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "tasks.json",
        help="저장 경로",
    )
    parser.add_argument(
        "--style",
        action="append",
        dest="styles",
        help="특정 스타일만 처리 (여러 번 지정 가능)",
    )
    parser.add_argument(
        "--max-scan-per-style",
        type=int,
        default=None,
        help="스타일당 최대 탐색 JSON 수 (테스트용)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="스타일 병렬 워커 수 (기본 12, 최대 스타일 수)",
    )
    parser.add_argument(
        "--json-workers",
        type=int,
        default=4,
        help="스타일 내 JSON 다운로드 병렬 수 (기본 4)",
    )
    args = parser.parse_args()

    hits, scanned = collect_stripe_indices(
        per_style=args.per_style,
        styles=args.styles,
        max_labels_per_style=args.max_scan_per_style,
        style_workers=args.workers,
        json_workers=args.json_workers,
    )

    styles_used = args.styles or STYLES
    payload = build_tasks_payload(
        hits,
        per_style=args.per_style,
        styles=styles_used,
        scanned_per_style=scanned,
    )
    save_tasks_json(payload, args.output)

    print(
        f"\n완료: {payload['actual_total']}건 수집 "
        f"(목표 {payload['target_total']}건)"
    )
    if payload["short_styles"]:
        print(f"부족 스타일 ({len(payload['short_styles'])}개): {', '.join(payload['short_styles'])}")
    print(f"저장: {args.output}")


if __name__ == "__main__":
    main()
