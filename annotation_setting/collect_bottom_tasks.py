"""
하의 샘플링 → tasks.json

- 라벨링 > 하의 가 있는 이미지만 대상
- 스타일당 드롭웨이스트(하의 > 디테일) 최대 20건
- 스타일당 그 외 하의 임의 최대 30건 (드롭웨이스트 샘플과 file_id 중복 없음)
- 스타일당 최대 50건 × 24스타일 = 최대 1200건

실행:
  python annotation_setting/collect_bottom_tasks.py
  python annotation_setting/collect_bottom_tasks.py --workers 12 --json-workers 8
"""
from __future__ import annotations

import argparse
import json
import random
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
from r2_io import R2Config, get_thread_client, iter_label_keys, load_json_from_r2  # noqa: E402

DROP_WAIST_DETAIL = "드롭웨이스트"
BOTTOM_TYPE = "하의"
DROP_WAIST_PER_STYLE = 20
ANY_BOTTOM_PER_STYLE = 30
MAX_PER_STYLE = DROP_WAIST_PER_STYLE + ANY_BOTTOM_PER_STYLE
MAX_TOTAL = MAX_PER_STYLE * len(STYLES)
RANDOM_SEED = 42


@dataclass
class BottomHit:
    file_id: int
    style: str
    label_key: str
    sample_group: str  # "drop_waist" | "any_bottom"


@dataclass
class StyleSampleResult:
    style: str
    hits: list[BottomHit]
    drop_waist_pool: int
    any_bottom_pool: int
    scanned: int
    elapsed_s: float
    error: str | None = None


def _labeling(raw: dict[str, Any]) -> dict:
    return (
        raw.get("데이터셋 정보", {})
        .get("데이터셋 상세설명", {})
        .get("라벨링", {})
    )


def _bottom_items(raw: dict[str, Any]) -> list[dict]:
    labeling = _labeling(raw)
    items = labeling.get(BOTTOM_TYPE, [])
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for item in items:
        if isinstance(item, dict) and item and item.get("카테고리"):
            out.append(item)
    return out


def _has_drop_waist_detail(raw: dict[str, Any]) -> bool:
    for item in _bottom_items(raw):
        details = item.get("디테일", [])
        if isinstance(details, str):
            details = [details]
        if DROP_WAIST_DETAIL in details:
            return True
    return False


def _has_any_bottom(raw: dict[str, Any]) -> bool:
    return len(_bottom_items(raw)) > 0


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


def _style_config(base: R2Config, style: str) -> R2Config:
    return R2Config(
        account_id=base.account_id,
        access_key=base.access_key,
        secret_key=base.secret_key,
        bucket=base.bucket,
        image_prefix=base.image_prefix,
        label_prefix=f"labeling/{style.strip('/')}/",
    )


def _classify_key(
    s3_client,
    style_config: R2Config,
    style: str,
    label_key: str,
) -> tuple[str, int, str] | None:
    """Returns (file_id, style, label_key) with group tag, or None."""
    try:
        raw = load_json_from_r2(s3_client, style_config, label_key)
    except Exception:
        return None

    if not _has_any_bottom(raw):
        return None

    file_id = _file_id_from_label(raw, label_key)
    if _has_drop_waist_detail(raw):
        return (file_id, style, label_key, "drop_waist")
    return (file_id, style, label_key, "any_bottom")


def _sample_style(
    drop_waist_pool: dict[int, tuple[str, str]],
    any_bottom_pool: dict[int, tuple[str, str]],
    *,
    rng: random.Random,
) -> list[BottomHit]:
    drop_ids = list(drop_waist_pool.keys())
    n_drop = min(DROP_WAIST_PER_STYLE, len(drop_ids))
    chosen_drop = rng.sample(drop_ids, n_drop) if n_drop else []

    drop_set = set(chosen_drop)
    any_candidates = [fid for fid in any_bottom_pool if fid not in drop_set]
    n_any = min(ANY_BOTTOM_PER_STYLE, len(any_candidates))
    chosen_any = rng.sample(any_candidates, n_any) if n_any else []

    hits: list[BottomHit] = []
    for fid in chosen_drop:
        style, key = drop_waist_pool[fid]
        hits.append(
            BottomHit(
                file_id=fid,
                style=style,
                label_key=key,
                sample_group="drop_waist",
            )
        )
    for fid in chosen_any:
        style, key = any_bottom_pool[fid]
        hits.append(
            BottomHit(
                file_id=fid,
                style=style,
                label_key=key,
                sample_group="any_bottom",
            )
        )
    return hits


def _collect_one_style(
    style: str,
    base_config: R2Config,
    *,
    json_workers: int,
    rng: random.Random,
) -> StyleSampleResult:
    t0 = time.perf_counter()
    style_config = _style_config(base_config, style)
    s3_client = get_thread_client(style_config)

    drop_waist_pool: dict[int, tuple[str, str]] = {}
    any_bottom_pool: dict[int, tuple[str, str]] = {}
    scanned = 0

    try:
        keys = list(iter_label_keys(s3_client, style_config))
        scanned = len(keys)

        def flush_classify(batch: list[str]) -> None:
            if json_workers <= 1:
                for key in batch:
                    row = _classify_key(s3_client, style_config, style, key)
                    if row is None:
                        continue
                    fid, st, k, group = row
                    if group == "drop_waist":
                        drop_waist_pool[fid] = (st, k)
                    else:
                        any_bottom_pool[fid] = (st, k)
                return

            with ThreadPoolExecutor(max_workers=json_workers) as pool:
                futures = [
                    pool.submit(_classify_key, s3_client, style_config, style, key)
                    for key in batch
                ]
                for fut in as_completed(futures):
                    row = fut.result()
                    if row is None:
                        continue
                    fid, st, k, group = row
                    if group == "drop_waist":
                        drop_waist_pool[fid] = (st, k)
                    else:
                        any_bottom_pool[fid] = (st, k)

        batch_size = max(16, json_workers * 8)
        for i in range(0, len(keys), batch_size):
            flush_classify(keys[i : i + batch_size])

        hits = _sample_style(drop_waist_pool, any_bottom_pool, rng=rng)

    except Exception as exc:
        return StyleSampleResult(
            style=style,
            hits=[],
            drop_waist_pool=0,
            any_bottom_pool=0,
            scanned=scanned,
            elapsed_s=time.perf_counter() - t0,
            error=str(exc),
        )

    return StyleSampleResult(
        style=style,
        hits=hits,
        drop_waist_pool=len(drop_waist_pool),
        any_bottom_pool=len(any_bottom_pool),
        scanned=scanned,
        elapsed_s=time.perf_counter() - t0,
    )


def collect_bottom_samples(
    *,
    styles: list[str] | None = None,
    style_workers: int = 12,
    json_workers: int = 8,
    seed: int = RANDOM_SEED,
) -> tuple[list[BottomHit], dict[str, dict[str, int]]]:
    base_config = R2Config.from_defaults()
    target_styles = styles or STYLES
    rng = random.Random(seed)
    style_rngs = {s: random.Random(rng.randint(0, 2**31 - 1)) for s in target_styles}

    max_workers = max(1, min(style_workers, len(target_styles)))
    print(
        f"[START] bottom sampling: styles={len(target_styles)}, "
        f"per_style=drop_waist<={DROP_WAIST_PER_STYLE}+any<={ANY_BOTTOM_PER_STYLE}, "
        f"max_total={MAX_TOTAL}, style_workers={max_workers}, json_workers={json_workers}"
    )

    results: dict[str, StyleSampleResult] = {}
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _collect_one_style,
                style,
                base_config,
                json_workers=json_workers,
                rng=style_rngs[style],
            ): style
            for style in target_styles
        }
        for fut in as_completed(futures):
            r = fut.result()
            results[r.style] = r
            if r.error:
                print(f"[ERROR] {r.style}: {r.error}")
                continue
            n_drop = sum(1 for h in r.hits if h.sample_group == "drop_waist")
            n_any = sum(1 for h in r.hits if h.sample_group == "any_bottom")
            print(
                f"[OK] {r.style}: tasks={len(r.hits)} "
                f"(drop_waist={n_drop}/{DROP_WAIST_PER_STYLE}, any={n_any}/{ANY_BOTTOM_PER_STYLE}, "
                f"pools drop={r.drop_waist_pool} any={r.any_bottom_pool}, scanned={r.scanned}, "
                f"{r.elapsed_s:.1f}s)"
            )

    hits: list[BottomHit] = []
    stats: dict[str, dict[str, int]] = {}
    for style in target_styles:
        r = results.get(style)
        if not r:
            continue
        hits.extend(r.hits)
        stats[style] = {
            "selected_drop_waist": sum(1 for h in r.hits if h.sample_group == "drop_waist"),
            "selected_any_bottom": sum(1 for h in r.hits if h.sample_group == "any_bottom"),
            "pool_drop_waist": r.drop_waist_pool,
            "pool_any_bottom": r.any_bottom_pool,
            "scanned_json": r.scanned,
        }

    print(f"[DONE] total_tasks={len(hits)}, elapsed={time.perf_counter() - t0:.1f}s")
    return hits, stats


def build_payload(hits: list[BottomHit], stats: dict[str, dict[str, int]]) -> dict[str, Any]:
    by_style: dict[str, list[int]] = {s: [] for s in STYLES}
    by_group: dict[str, list[int]] = {"drop_waist": [], "any_bottom": []}

    for hit in hits:
        by_style.setdefault(hit.style, []).append(hit.file_id)
        by_group.setdefault(hit.sample_group, []).append(hit.file_id)

    short_styles = [
        s
        for s, st in stats.items()
        if st["selected_drop_waist"] + st["selected_any_bottom"] < MAX_PER_STYLE
        and (st["pool_drop_waist"] < DROP_WAIST_PER_STYLE or st["pool_any_bottom"] < ANY_BOTTOM_PER_STYLE)
    ]

    return {
        "source": "bottom_sampling",
        "clothing_type": BOTTOM_TYPE,
        "sampling": {
            "drop_waist_detail": DROP_WAIST_DETAIL,
            "drop_waist_per_style_max": DROP_WAIST_PER_STYLE,
            "any_bottom_per_style_max": ANY_BOTTOM_PER_STYLE,
            "max_per_style": MAX_PER_STYLE,
            "max_total": MAX_TOTAL,
            "random_seed": RANDOM_SEED,
        },
        "filter": {
            "clothing_type": BOTTOM_TYPE,
            "drop_waist_detail": DROP_WAIST_DETAIL,
        },
        "target_total": MAX_TOTAL,
        "actual_total": len(hits),
        "indices": [h.file_id for h in hits],
        "by_style": by_style,
        "by_sample_group": by_group,
        "records": [asdict(h) for h in hits],
        "per_style_stats": stats,
        "short_styles": short_styles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="하의 드롭웨이스트+임의 샘플링 → tasks.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR / "tasks.json",
    )
    parser.add_argument("--style", action="append", dest="styles")
    parser.add_argument("--workers", type=int, default=12, help="스타일 병렬 수 (기본 12)")
    parser.add_argument("--json-workers", type=int, default=8, help="스타일 내 JSON 병렬 수 (기본 8)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    hits, stats = collect_bottom_samples(
        styles=args.styles,
        style_workers=args.workers,
        json_workers=args.json_workers,
        seed=args.seed,
    )
    payload = build_payload(hits, stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    n_drop = len(payload["by_sample_group"]["drop_waist"])
    n_any = len(payload["by_sample_group"]["any_bottom"])
    print(
        f"\n완료: {payload['actual_total']}건 "
        f"(drop_waist={n_drop}, any_bottom={n_any}, 목표<={MAX_TOTAL})"
    )
    print(f"저장: {args.output}")
    if payload["short_styles"]:
        print(f"부족 스타일: {', '.join(payload['short_styles'])}")


if __name__ == "__main__":
    main()
