# 마스크 데이터를 R2에서 다운로드하여 폴리곤 마스크 생성하는 스크립트
from __future__ import annotations

import argparse
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
VLM_DIR = REPO_ROOT / "VLM_captioning"
if str(VLM_DIR) not in sys.path:
    sys.path.insert(0, str(VLM_DIR))

from vlm_common import (
    R2Config,
    create_r2_client,
    iter_label_keys,
    load_image_from_r2,
    load_json_from_r2,
    resolve_image_key_from_label,
)  # noqa: E402


CATEGORY_MAP = {
    "아우터": "outerwear",
    "원피스": "dress",
    "상의": "top",
    "하의": "bottom",
}
MAX_POLYGON_POINTS = 64
_THREAD_LOCAL = threading.local()


def _extract_polygon_points(polygon_item: dict[str, Any]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for idx in range(1, MAX_POLYGON_POINTS + 1):
        x_key = f"X좌표{idx}"
        y_key = f"Y좌표{idx}"
        if x_key not in polygon_item or y_key not in polygon_item:
            break
        points.append((int(polygon_item[x_key]), int(polygon_item[y_key])))
    return points


def _apply_multi_polygon_mask(
    image: Image.Image,
    polygon_items: list[dict[str, Any]],
    *,
    crop: bool = True,
) -> Image.Image | None:
    mask = Image.new("L", image.size, 0)
    drawer = ImageDraw.Draw(mask)
    valid_polygon_count = 0

    for polygon_item in polygon_items:
        if not polygon_item:
            continue
        points = _extract_polygon_points(polygon_item)
        if len(points) < 3:
            continue
        drawer.polygon(points, outline=1, fill=1)
        valid_polygon_count += 1

    if valid_polygon_count == 0:
        return None

    mask_np = np.array(mask, dtype=bool)
    if not mask_np.any():
        return None

    image_np = np.array(image)
    result = np.zeros_like(image_np)
    result[mask_np] = image_np[mask_np]

    if not crop:
        return Image.fromarray(result)

    ys, xs = np.where(mask_np)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    cropped = result[y_min : y_max + 1, x_min : x_max + 1]
    return Image.fromarray(cropped)


def _image_id_from_label(raw: dict[str, Any], label_key: str) -> str:
    image_info = raw.get("이미지 정보", {})
    dataset_info = raw.get("데이터셋 정보", {})

    image_id = image_info.get("이미지 식별자")
    if image_id not in (None, ""):
        return str(image_id)

    file_number = dataset_info.get("파일 번호")
    if file_number not in (None, ""):
        return str(file_number)

    return Path(label_key).stem


def _write_checkpoint(checkpoint_file: Path | None, label_key: str) -> None:
    if checkpoint_file is None:
        return
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file.write_text(label_key, encoding="utf-8")


def _iter_filtered_label_keys(
    s3_client,
    config: R2Config,
    *,
    style: str | None = None,
    label_prefix: str | None = None,
    start_after_key: str | None = None,
) -> Any:
    if label_prefix:
        prefix = label_prefix.strip("/")
        config.label_prefix = f"{prefix}/"
    elif style:
        config.label_prefix = f"labeling/{style.strip('/')}/"
    else:
        config.label_prefix = "labeling/"

    for label_key in iter_label_keys(s3_client, config):
        if start_after_key and label_key <= start_after_key:
            continue
        yield label_key


def _get_thread_client(config: R2Config):
    if not hasattr(_THREAD_LOCAL, "s3_client"):
        _THREAD_LOCAL.s3_client = create_r2_client(config)
    return _THREAD_LOCAL.s3_client


def _process_label_key(
    label_key: str,
    *,
    config: R2Config,
    output_dir: Path,
    crop: bool,
) -> tuple[str, int, str]:
    """
    반환: (label_key, saved_count, status)
    status: "ok" | "skip"
    """
    s3_client = _get_thread_client(config)
    try:
        raw = load_json_from_r2(s3_client, config, label_key)
        image_key = resolve_image_key_from_label(s3_client, config, label_key, raw)
        image = load_image_from_r2(s3_client, config, image_key)
    except Exception as exc:
        print(f"[SKIP] {label_key}: {exc}")
        return label_key, 0, "skip"

    polygon_root = (
        raw.get("데이터셋 정보", {}).get("데이터셋 상세설명", {}).get("폴리곤좌표", {})
    )
    image_id = _image_id_from_label(raw, label_key)

    one_image_saved = 0
    for category_ko, category_en in CATEGORY_MAP.items():
        polygon_items = polygon_root.get(category_ko, [])
        if not isinstance(polygon_items, list):
            continue

        masked = _apply_multi_polygon_mask(image, polygon_items, crop=crop)
        if masked is None:
            continue

        out_path = output_dir / f"{image_id}_{category_en}.jpg"
        masked.convert("RGB").save(out_path, format="JPEG", quality=95)
        one_image_saved += 1

    print(f"[OK] {label_key} -> {one_image_saved} file(s)")
    return label_key, one_image_saved, "ok"


def generate_masks(
    *,
    max_labels: int | None = None,
    output_dir: Path,
    crop: bool = True,
    workers: int = 1,
    style: str | None = None,
    label_prefix: str | None = None,
    start_after_key: str | None = None,
    checkpoint_file: Path | None = None,
) -> None:
    config = R2Config.from_env()
    # 사용자 요구사항 기본값 보정
    config.bucket = "project2"
    config.image_prefix = "image/"

    s3_client = create_r2_client(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    ok_count = 0
    skip_count = 0
    saved_count = 0
    max_workers = max(1, workers)
    max_in_flight = max_workers * 4

    print(
        "[START] "
        f"workers={max_workers}, bucket={config.bucket}, "
        f"label_prefix={label_prefix or (f'labeling/{style}/' if style else 'labeling/')}, "
        f"start_after_key={start_after_key or '-'}, max_labels={max_labels or 'all'}"
    )

    key_iter = _iter_filtered_label_keys(
        s3_client,
        config,
        style=style,
        label_prefix=label_prefix,
        start_after_key=start_after_key,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight: dict[Future, str] = {}

        def submit_until_full() -> bool:
            nonlocal processed_count
            while len(in_flight) < max_in_flight:
                if max_labels is not None and processed_count + len(in_flight) >= max_labels:
                    return False
                try:
                    label_key = next(key_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    _process_label_key,
                    label_key,
                    config=config,
                    output_dir=output_dir,
                    crop=crop,
                )
                in_flight[future] = label_key
            return True

        submit_until_full()

        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                label_key = in_flight.pop(fut)
                try:
                    _, one_saved, status = fut.result()
                except Exception as exc:
                    print(f"[SKIP] {label_key}: {exc}")
                    one_saved = 0
                    status = "skip"

                processed_count += 1
                saved_count += one_saved
                if status == "ok":
                    ok_count += 1
                else:
                    skip_count += 1
                _write_checkpoint(checkpoint_file, label_key)

            if max_labels is not None and processed_count >= max_labels:
                for fut in in_flight:
                    fut.cancel()
                break

            submit_until_full()

    print(
        f"\n완료: 라벨 {processed_count}건 처리 (성공 {ok_count}, 스킵 {skip_count}), "
        f"마스크 {saved_count}장 저장"
    )
    print(f"저장 경로: {output_dir}")
    if checkpoint_file:
        print(f"체크포인트: {checkpoint_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cloudflare R2(project2) image/labeling 폴리곤 마스크 생성"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "masking_data",
        help="마스크 이미지 저장 폴더",
    )
    parser.add_argument("--limit", type=int, default=None, help="(호환) 처리 개수 제한")
    parser.add_argument("--max-labels", type=int, default=None, help="처리할 라벨 JSON 개수 제한")
    parser.add_argument("--workers", type=int, default=1, help="병렬 워커 수 (기본 1)")
    parser.add_argument("--style", type=str, default=None, help="특정 스타일만 처리 (예: classic)")
    parser.add_argument(
        "--label-prefix",
        type=str,
        default=None,
        help="직접 라벨 prefix 지정 (예: labeling/classic/)",
    )
    parser.add_argument(
        "--start-after-key",
        type=str,
        default=None,
        help="해당 key 이후부터 처리 재개 (예: labeling/classic/1032711.json)",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=Path(__file__).resolve().parent / "masking_data" / "last_checkpoint.txt",
        help="마지막 처리 key 저장 파일",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="폴리곤 bbox crop 비활성화",
    )
    args = parser.parse_args()

    effective_max_labels = args.max_labels if args.max_labels is not None else args.limit

    generate_masks(
        max_labels=effective_max_labels,
        output_dir=args.output_dir,
        crop=not args.no_crop,
        workers=args.workers,
        style=args.style,
        label_prefix=args.label_prefix,
        start_after_key=args.start_after_key,
        checkpoint_file=args.checkpoint_file,
    )


if __name__ == "__main__":
    main()
