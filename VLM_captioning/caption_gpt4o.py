"""
Cloudflare R2 K-fashion 이미지 → GPT-4o 상세 캡셔닝.

사용 예:
  set OPENAI_API_KEY=sk-...
  python caption_gpt4o.py --limit 5
  python caption_gpt4o.py --mask --keys 11.json 12101.json
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys

from openai import OpenAI
from PIL import Image

from caption_pipeline import add_common_args, run_caption_pipeline

MODEL_ID = "gpt-4o"
MODEL_SLUG = "gpt4o"


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


def run(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY 환경 변수가 필요합니다.")
        return 1

    client = OpenAI(api_key=api_key)

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
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    sys.exit(run(cli_args))
