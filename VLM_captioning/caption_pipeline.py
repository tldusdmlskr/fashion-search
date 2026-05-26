"""VLM 캡셔닝 공통 실행 루프 (원본 / 마스킹 모드)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_common import (
    CaptionItem,
    R2Config,
    build_prompt,
    build_result_record,
    create_r2_client,
    default_output_path,
    iter_caption_items,
    parse_json_response,
    save_results_jsonl,
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=None, help="처리할 항목 수 제한")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="이미지/JSON 파일명 또는 R2 key (예: 11.jpg, 11.json)",
    )
    parser.add_argument("--output", type=str, default=None, help="출력 JSONL 경로")
    parser.add_argument(
        "--mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="R2 폴리곤 마스킹 후 요소별 캡셔닝 (기본: 켜짐). --no-mask 로 원본 이미지",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="마스킹 후 bbox crop 비활성화 (전체 프레임 유지)",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="마스킹 대상 카테고리 (기본: 상의, 하의, 아우터, 원피스)",
    )


def resolve_output_path(args: argparse.Namespace, model_slug: str) -> Path:
    if args.output:
        return Path(args.output)
    slug = f"{model_slug}_masked" if args.mask else model_slug
    return default_output_path(slug)


def run_caption_pipeline(
    args: argparse.Namespace,
    *,
    model_slug: str,
    model_name: str,
    caption_fn: Callable[[Image.Image, str], str],
    prompt_builder: Callable[[CaptionItem | None], str] | None = None,
) -> int:
    config = R2Config.from_env()
    s3 = create_r2_client(config)

    items = list(
        iter_caption_items(
            s3,
            config,
            use_mask=args.mask,
            crop=not args.no_crop,
            limit=args.limit,
            keys=args.keys,
            categories=args.categories,
        )
    )

    if not items:
        if args.mask:
            print(
                f"No label polygons found under s3://{config.bucket}/{config.label_prefix}"
            )
        else:
            print(f"No images found under s3://{config.bucket}/{config.image_prefix}")
        return 1

    mode = "masked_polygon" if args.mask else "original"
    print(f"Processing {len(items)} region(s) [{mode}] with {model_name}...")

    build_prompt_fn = prompt_builder or build_prompt
    records: list[dict[str, Any]] = []

    for item in items:
        label = f"  - {item.display_name}"
        print(label)
        try:
            raw = caption_fn(item.image, build_prompt_fn(item))
            caption = parse_json_response(raw)
            records.append(
                build_result_record(
                    model_name=model_name,
                    image_key=item.image_key,
                    caption=caption,
                    raw_response=raw,
                    masked=item.masked,
                    label_key=item.label_key,
                    garment_category=item.garment_category,
                    polygon_index=item.polygon_index,
                )
            )
        except Exception as exc:
            records.append(
                build_result_record(
                    model_name=model_name,
                    image_key=item.image_key,
                    caption=None,
                    raw_response="",
                    error=str(exc),
                    masked=item.masked,
                    label_key=item.label_key,
                    garment_category=item.garment_category,
                    polygon_index=item.polygon_index,
                )
            )
            print(f"    error: {exc}")

    out_path = resolve_output_path(args, model_slug)
    save_results_jsonl(records, out_path)
    print(f"Saved {len(records)} record(s) → {out_path}")
    return 0
