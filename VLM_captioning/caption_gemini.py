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
import os
import sys
import time

import google.generativeai as genai
from PIL import Image

from caption_pipeline import add_common_args, run_caption_pipeline
from vlm_common import build_gemini_element_prompt

MODEL_SLUG = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

# 환경 변수 없을 때 사용 (generate_caption.py와 동일)
DEFAULT_API_KEY = "AIzaSyBv5s7dVv4VEFb5rOam4yQgl7HHd3fJ5no"


def caption_image(model, image: Image.Image, prompt: str) -> str:
    response = model.generate_content(
        [prompt, image],
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 800,
        },
    )
    return response.text or ""


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
        "--model",
        type=str,
        default=None,
        help=f"Gemini 모델 ID (기본: {DEFAULT_GEMINI_MODEL})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    sys.exit(run(cli_args))
