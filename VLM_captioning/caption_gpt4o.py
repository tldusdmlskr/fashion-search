"""
Cloudflare R2 K-fashion 이미지 → GPT-4o 상세 캡셔닝.

사용 예:
  set OPENAI_API_KEY=sk-...
  python caption_gpt4o.py --limit 5
  python caption_gpt4o.py --mask --keys 11.json 12101.json
  python caption_gpt4o.py --local-dir "..\\test_data" --keys 11.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from dotenv import dotenv_values, load_dotenv
from openai import OpenAI
from PIL import Image

from caption_pipeline import add_common_args, run_caption_pipeline
from vlm_common import (
    CaptionItem,
    GARMENT_CATEGORIES,
    apply_polygon_mask,
    build_gemini_element_prompt,
    build_result_record,
    default_output_path,
    extract_polygon_points,
    parse_json_response,
    save_results_jsonl,
)

MODEL_ID = "gpt-4o"
MODEL_SLUG = "gpt4o"

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE / ".env.example")
_ENV_EXAMPLE = dotenv_values(_HERE / ".env.example")


def image_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}"


def caption_image(client: OpenAI, image: Image.Image, prompt: str) -> str:
    data_url = image_to_data_url(image)

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        max_tokens=800,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _iter_local_items(
    local_dir: Path,
    *,
    limit: int | None,
    keys: list[str] | None,
    crop: bool,
    categories: list[str] | None,
) -> Iterator[CaptionItem]:
    target_categories = categories or GARMENT_CATEGORIES

    if keys:
        json_files = [local_dir / k for k in keys]
    else:
        json_files = sorted(local_dir.glob("*.json"))

    produced = 0

    for json_path in json_files:
        if not json_path.exists():
            continue

        raw = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = local_dir / f"{json_path.stem}.jpg"
        if not image_path.exists():
            for ext in (".jpeg", ".png", ".webp"):
                alt = local_dir / f"{json_path.stem}{ext}"
                if alt.exists():
                    image_path = alt
                    break
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        polygon_root = (
            (raw.get("데이터셋 정보", {}) or {})
            .get("데이터셋 상세설명", {})
            .get("폴리곤좌표", {})
        )

        for category in target_categories:
            items = polygon_root.get(category, [])
            if not isinstance(items, list):
                continue
            for polygon_index, polygon_item in enumerate(items):
                if not polygon_item:
                    continue
                points = extract_polygon_points(polygon_item)
                if len(points) < 3:
                    continue

                masked = apply_polygon_mask(image, points, crop=crop)
                yield CaptionItem(
                    image_key=f"local:{image_path.name}",
                    image=masked,
                    masked=True,
                    label_key=f"local:{json_path.name}",
                    garment_category=category,
                    polygon_index=polygon_index,
                )
                produced += 1
                if limit is not None and produced >= limit:
                    return


def _run_local(args: argparse.Namespace, client: OpenAI) -> int:
    local_dir = Path(args.local_dir).resolve()
    crop = not args.no_crop
    items = list(
        _iter_local_items(
            local_dir,
            limit=args.limit,
            keys=args.keys,
            crop=crop,
            categories=args.categories,
        )
    )
    if not items:
        print(f"로컬에서 처리할 항목이 없습니다: {local_dir}")
        return 1

    records: list[dict[str, Any]] = []
    for item in items:
        print(f"  - {item.display_name}")
        try:
            prompt = build_gemini_element_prompt(item)
            raw = caption_image(client, item.image, prompt)
            caption = parse_json_response(raw)
            records.append(
                build_result_record(
                    model_name=MODEL_ID,
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
                    model_name=MODEL_ID,
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
        time.sleep(0.3)

    out_path = Path(args.output) if args.output else default_output_path("gpt4o_local_masked")
    save_results_jsonl(records, out_path)
    print(f"Saved {len(records)} record(s) → {out_path}")
    return 0


def run(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "") or _ENV_EXAMPLE.get("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY 환경 변수가 필요합니다.")
        return 1

    client = OpenAI(api_key=api_key)

    if getattr(args, "local_dir", None):
        return _run_local(args, client)

    def _caption(image: Image.Image, prompt: str) -> str:
        return caption_image(client, image, prompt)

    return run_caption_pipeline(
        args,
        model_slug=MODEL_SLUG,
        model_name=MODEL_ID,
        caption_fn=_caption,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-4o K-fashion captioning from R2")
    add_common_args(parser)
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="R2 대신 로컬 폴더(jpg+json)에서 실행 (예: ..\\test_data)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    sys.exit(run(cli_args))
