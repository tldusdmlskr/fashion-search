"""
Cloudflare R2 K-fashion → Gemini 요소별 캡셔닝 (폴리곤 마스킹 기본).

폴리곤 외부 배경 제거 → bbox crop → Gemini (generate_caption.py와 동일 프롬프트·출력).

사용 예:
  python caption_gemini.py --limit 5
  python caption_gemini.py --keys 11.json
  python caption_gemini.py --no-mask --keys 11.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import google.generativeai as genai
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

MODEL_SLUG = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

# 환경 변수 없을 때 사용 (generate_caption.py와 동일)
DEFAULT_API_KEY = "AIzaSyA0Lz_vaci9VtyI-Go7_4ZaXbb6xMoXaUw"


def caption_image(model, image: Image.Image, prompt: str) -> str:
    response = model.generate_content(
        [prompt, image],
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 800,
        },
    )
    return response.text or ""


def _iter_local_items(
    local_dir: Path,
    *,
    limit: int | None,
    keys: list[str] | None,
    crop: bool,
    categories: list[str] | None,
) -> Iterator[CaptionItem]:
    """로컬 test_data의 jpg+json에서 폴리곤 마스킹 항목 생성."""
    target_categories = categories or GARMENT_CATEGORIES

    def _load_json_auto(path: Path) -> dict[str, Any]:
        raw_bytes = path.read_bytes()
        for enc in ("utf-8", "utf-8-sig", "cp949"):
            try:
                obj = json.loads(raw_bytes.decode(enc))
                # 키가 깨진(� 포함) 경우 cp949로 재시도 유도
                if "이미지 정보" in obj and "데이터셋 정보" in obj:
                    return obj
                if enc == "cp949":
                    return obj
            except Exception:
                continue
        # 마지막 fallback
        return json.loads(raw_bytes.decode("cp949", errors="replace"))

    def _get(obj: dict[str, Any], *candidates: str, default=None):
        for k in candidates:
            if isinstance(obj, dict) and k in obj:
                return obj[k]
        return default

    if keys:
        json_files = [local_dir / k for k in keys]
    else:
        json_files = sorted(local_dir.glob("*.json"))

    produced = 0

    for json_path in json_files:
        if not json_path.exists():
            continue

        raw = _load_json_auto(json_path)
        # 일부 test_data JSON은 키가 깨져 저장된 케이스가 있어 후보 키 모두 지원
        image_info = _get(raw, "이미지 정보", "�̹��� ����", default={}) or {}
        image_file = _get(image_info, "이미지 파일명", "�̹��� ������", default="") or ""
        image_path: Path | None = None
        if image_file:
            candidate = local_dir / Path(image_file).name
            if candidate.exists():
                image_path = candidate

        if image_path is None:
            candidate = local_dir / f"{json_path.stem}.jpg"
            if candidate.exists():
                image_path = candidate
            else:
                for ext in (".jpeg", ".png", ".webp"):
                    alt = local_dir / f"{json_path.stem}{ext}"
                    if alt.exists():
                        image_path = alt
                        break

        if image_path is None or not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        dataset_info = _get(raw, "데이터셋 정보", "�����ͼ� ����", default={}) or {}
        detail = _get(dataset_info, "데이터셋 상세설명", "�����ͼ� �󼼼���", default={}) or {}
        polygon_root = _get(detail, "폴리곤좌표", "��������ǥ", default={}) or {}

        category_alias = {
            "아우터": "�ƿ���",
            "하의": "����",
            "원피스": "���ǽ�",
            "상의": "����",
        }

        for category in target_categories:
            items = polygon_root.get(category, None)
            if items is None:
                items = polygon_root.get(category_alias.get(category, ""), [])
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


def _run_local(args: argparse.Namespace, gemini_model, model_id: str) -> int:
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
            raw_text = caption_image(
                gemini_model, item.image, build_gemini_element_prompt(item)
            )
            caption = parse_json_response(raw_text)
            records.append(
                build_result_record(
                    model_name=model_id,
                    image_key=item.image_key,
                    caption=caption,
                    raw_response=raw_text,
                    masked=item.masked,
                    label_key=item.label_key,
                    garment_category=item.garment_category,
                    polygon_index=item.polygon_index,
                )
            )
        except Exception as exc:
            records.append(
                build_result_record(
                    model_name=model_id,
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

    out_path = Path(args.output) if args.output else default_output_path("gemini_local_masked")
    save_results_jsonl(records, out_path)
    print(f"Saved {len(records)} record(s) → {out_path}")
    return 0


def run(args: argparse.Namespace) -> int:
    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or DEFAULT_API_KEY
    )
    if not api_key:
        print("GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경 변수가 필요합니다.")
        return 1

    model_id = args.model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model_id)

    if getattr(args, "local_dir", None):
        return _run_local(args, gemini_model, model_id)

    def _caption(image: Image.Image, prompt: str) -> str:
        return caption_image(gemini_model, image, prompt)

    result = run_caption_pipeline(
        args,
        model_slug=(
            MODEL_SLUG
            if model_id == DEFAULT_GEMINI_MODEL
            else model_id.replace("/", "_")
        ),
        model_name=model_id,
        caption_fn=_caption,
        prompt_builder=build_gemini_element_prompt,
    )

    # API rate limit 여유
    if result == 0:
        time.sleep(0.1)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini 폴리곤 마스킹 요소별 K-fashion 캡셔닝 (R2)"
    )
    add_common_args(parser)
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="R2 대신 로컬 폴더(jpg+json)에서 실행 (예: ..\\test_data). keys는 json 파일명으로 해석",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Gemini 모델 ID (기본: {DEFAULT_GEMINI_MODEL})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    sys.exit(run(cli_args))
