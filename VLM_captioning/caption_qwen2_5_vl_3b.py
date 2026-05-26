"""
Cloudflare R2 K-fashion 이미지 → Qwen2.5-VL-3B-Instruct 상세 캡셔닝.

사용 예:
  python caption_qwen2_5_vl_3b.py --limit 5
  python caption_qwen2_5_vl_3b.py --mask --keys 11.json
  python caption_qwen2_5_vl_3b.py --mask --categories 상의 --limit 10
"""

from __future__ import annotations

import argparse
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from caption_pipeline import add_common_args, run_caption_pipeline

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_SLUG = "qwen2_5_vl_3b"

_model = None
_processor = None


def load_model(device: str | None = None):
    global _model, _processor
    if _model is not None:
        return _model, _processor

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device is None else None,
    )
    if device is not None:
        _model = _model.to(device)
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    return _model, _processor


def caption_image(model, processor, image: Image.Image, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
    )[0]


def run(args: argparse.Namespace) -> int:
    print(f"Loading model: {MODEL_ID}")
    model, processor = load_model()

    def _caption(image: Image.Image, prompt: str) -> str:
        return caption_image(model, processor, image, prompt)

    return run_caption_pipeline(
        args,
        model_slug=MODEL_SLUG,
        model_name=MODEL_ID,
        caption_fn=_caption,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL-3B K-fashion captioning from R2"
    )
    add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    sys.exit(run(cli_args))
