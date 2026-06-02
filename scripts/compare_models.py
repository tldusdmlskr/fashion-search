"""
compare_models.py — GPT-4o mini vs Gemini 2.5 Flash 캡션 품질 비교
실행: python scripts/compare_models.py

test_data/ 폴더의 이미지에 폴리곤 마스킹 적용 후 두 모델에 동일 입력으로 비교.
결과: test_data/compare_results.jsonl
"""
import sys
import json
import base64
import io
import asyncio
import time
from pathlib import Path

import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.caption import SYSTEM_PROMPT, _make_config, CaptionOutput
from src.masking import load_masked_items

TEST_DIR  = Path(__file__).parent.parent / "test_data"
OUT_PATH  = TEST_DIR / "compare_results.jsonl"


def make_masked_image(jpg_path: Path, json_path: Path) -> list[tuple[str, Image.Image]]:
    return [(cat, img) for cat, _, img in load_masked_items(json_path, jpg_path)]


def pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ── GPT-4o mini ──────────────────────────────────────────────────────────────

def run_gpt4o_mini(image: Image.Image) -> dict:
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    b64    = pil_to_b64(image)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            }
        ],
        max_tokens=300,
    )
    raw  = resp.choices[0].message.content
    result = json.loads(raw)
    usage  = resp.usage
    return {
        "output":       result,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost_usd":     usage.prompt_tokens / 1e6 * 0.15 + usage.completion_tokens / 1e6 * 0.60,
    }


# ── Gemini 2.5 Flash ─────────────────────────────────────────────────────────

def run_gemini_flash(image: Image.Image) -> dict:
    from google import genai
    from google.genai import types

    client   = genai.Client(api_key=GEMINI_API_KEY)
    img_bytes = pil_to_bytes(image)

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            SYSTEM_PROMPT,
        ],
        config=_make_config(),
    )
    result = json.loads(resp.text)
    meta   = resp.usage_metadata
    input_tok  = meta.prompt_token_count if meta else 0
    output_tok = meta.candidates_token_count if meta else 0
    return {
        "output":        result,
        "input_tokens":  input_tok,
        "output_tokens": output_tok,
        "cost_usd":      input_tok / 1e6 * 0.15 + output_tok / 1e6 * 0.60,
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    import os
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if not has_openai:
        print("⚠ OPENAI_API_KEY 없음 — Gemini만 실행")

    jpg_files = sorted(TEST_DIR.glob("*.jpg"))
    print(f"테스트 이미지: {len(jpg_files)}개\n")

    all_results = []
    totals = {
        "gpt4o_mini": {"input": 0, "output": 0, "cost": 0.0, "n": 0},
        "gemini_flash": {"input": 0, "output": 0, "cost": 0.0, "n": 0},
    }

    for jpg_path in jpg_files:
        json_path = jpg_path.with_suffix(".json")
        if not json_path.exists():
            continue

        masked_list = make_masked_image(jpg_path, json_path)
        if not masked_list:
            print(f"  {jpg_path.name}: 폴리곤 없음, 스킵")
            continue

        for cat, masked_img in masked_list:
            label = f"{jpg_path.stem} [{cat}]"
            print(f"{'─'*60}")
            print(f"  {label}")

            row = {"file": jpg_path.stem, "category": cat}

            # Gemini
            try:
                t0 = time.time()
                g  = run_gemini_flash(masked_img)
                elapsed = time.time() - t0
                row["gemini"] = g["output"]
                row["gemini_tokens"] = {"in": g["input_tokens"], "out": g["output_tokens"]}
                row["gemini_cost_usd"] = g["cost_usd"]
                totals["gemini_flash"]["input"]  += g["input_tokens"]
                totals["gemini_flash"]["output"] += g["output_tokens"]
                totals["gemini_flash"]["cost"]   += g["cost_usd"]
                totals["gemini_flash"]["n"]      += 1
                print(f"  [Gemini 2.5 Flash] {elapsed:.1f}s  ${g['cost_usd']:.5f}")
                print(f"    category     : {g['output'].get('category','')}")
                print(f"    micro_details: {g['output'].get('micro_details','')}")
                print(f"    mood_and_tpo : {g['output'].get('mood_and_tpo','')}")
            except Exception as e:
                row["gemini"] = {"error": str(e)}
                print(f"  [Gemini] ERROR: {e}")

            # GPT-4o mini
            if has_openai:
                try:
                    t0 = time.time()
                    gp = run_gpt4o_mini(masked_img)
                    elapsed = time.time() - t0
                    row["gpt4o_mini"] = gp["output"]
                    row["gpt4o_mini_tokens"] = {"in": gp["input_tokens"], "out": gp["output_tokens"]}
                    row["gpt4o_mini_cost_usd"] = gp["cost_usd"]
                    totals["gpt4o_mini"]["input"]  += gp["input_tokens"]
                    totals["gpt4o_mini"]["output"] += gp["output_tokens"]
                    totals["gpt4o_mini"]["cost"]   += gp["cost_usd"]
                    totals["gpt4o_mini"]["n"]      += 1
                    print(f"  [GPT-4o mini]      {elapsed:.1f}s  ${gp['cost_usd']:.5f}")
                    print(f"    category     : {gp['output'].get('category','')}")
                    print(f"    micro_details: {gp['output'].get('micro_details','')}")
                    print(f"    mood_and_tpo : {gp['output'].get('mood_and_tpo','')}")
                except Exception as e:
                    row["gpt4o_mini"] = {"error": str(e)}
                    print(f"  [GPT-4o mini] ERROR: {e}")

            all_results.append(row)

    # 저장
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 비용 요약
    n_items = 100_000
    print(f"\n{'='*60}")
    print(f"실측 토큰 기반 10만 건 예상 비용 (샘플 {totals['gemini_flash']['n']}건 기준)")
    print(f"{'='*60}")
    for model, t in totals.items():
        if t["n"] == 0:
            continue
        scale   = n_items / t["n"]
        est_usd = t["cost"] * scale
        est_krw = est_usd * 1400
        avg_in  = t["input"] / t["n"]
        avg_out = t["output"] / t["n"]
        print(f"  {model:<18}  avg {avg_in:.0f}in/{avg_out:.0f}out tok"
              f"  -> 10만건 예상 ${est_usd:.1f}  (약 {est_krw:,.0f}원)")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
