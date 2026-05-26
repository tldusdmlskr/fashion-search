"""
Gemini 캡셔닝 — Cloudflare R2 폴리곤 마스킹 후 요소(상의/하의/아우터/원피스)별 캡션.

노트북(VLM 태깅 검증_마스킹 데이터.ipynb)과 동일: 폴리곤 외부는 검정 처리 → bbox crop → VLM 입력.

사용 예:
  python generate_caption.py --url "https://.../image/modern/1106011.jpg"
  python generate_caption.py --batch tasks.json --out tasks_captioned.json --limit 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import google.generativeai as genai
import requests
from io import BytesIO
from PIL import Image

from vlm_common import (
    GARMENT_CATEGORIES,
    R2Config,
    apply_polygon_mask,
    create_r2_client,
    extract_polygon_points,
    load_image_from_r2,
    load_json_from_r2,
    parse_json_response,
    resolve_image_key_from_label,
    resolve_label_key_from_image,
)

API_KEY = "AIzaSyBv5s7dVv4VEFb5rOam4yQgl7HHd3fJ5no"
MODEL = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = """역할:
당신은 한국 이커머스 패션 플랫폼의 전문 카탈로그 데이터 구축 모델입니다.

과제:
입력 이미지는 착용샷에서 폴리곤으로 분리·배경 제거된 단일 의류 영역입니다.
(1) 보완 물리 디테일 캡션과 (2) 감성·TPO 태그를 JSON으로 추출하십시오.
배경·얼굴·다른 의류 영역은 이미 제거되었습니다. 보이는 의류 픽셀만 분석하십시오.

[출력 금지 — K-Fashion 데이터셋·JSON에 이미 존재하는 속성. 추측·서술·dense_caption 반복 금지]
(출처: K-Fashion 이미지 데이터 구축 가이드라인 — 11개 상위 속성, 스타일 23종, 세부속성 186종)

■ 스타일(다중 레이블, 상위 10 / 하위 23 예시)
  트래디셔널: 클래식, 프레피 | 매니시: 매니시, 톰보이 | 페미닌: 엘레강스, 로맨틱, 섹시
  에스닉: 히피, 웨스턴, 오리엔탈 | 컨템포러리: 모던, 미니멀, 소피스트케이티드, 아방가르드
  내추럴: 컨트리, 리조트 | 젠더리스 | 스포티 | 문화/서브컬처: 레트로, 키치·키덜트, 힙합, 펑크
  캐주얼: 캐주얼, 밀리터리, 스트리트, 놈코어, 맥시멈 | 애슬레저 등

■ 카테고리(대분류 4 + 세부 아이템)
  상의: 탑, 블라우스, 티셔츠, 니트웨어, 셔츠, 브라탑, 후드티, 캐주얼상의
  하의: 청바지, 팬츠, 스커트, 레깅스, 조거팬츠
  아우터: 코트, 재킷, 점퍼, 패딩, 베스트, 가디건, 짚업
  원피스: 드레스, 점프수트, 수영복 등

■ 컬러: 블랙, 화이트, 그레이, 레드, 핑크, 오렌지, 베이지, 브라운, 옐로우, 그린, 카키, 네이비, 블루, 스카이블루, 퍼플, 라벤더, 와인, 골드, 네온, 민트 등

■ 디테일: 비즈, 단추, 니트꽈배기, 체인, 컷오프, 더블브레스티드, 드롭숄더, 자수, 프릴, 프린지, 퍼프, 후드, 패딩, 패치워크, 플리츠, 포켓, 지퍼업, 퀼팅, 컷아웃, 스티치, 슬릿, 스터드, 디스트로이드, 셔링 등

■ 프린트: 체크, 스트라이프, 지그재그, 호피, 지브라, 도트, 무지, 카무플라쥬, 그래픽, 레터링, 페이즐리, 하운즈투스, 아가일, 깅엄, 플로럴, 타이다이, 그라데이션, 해골, 기하학 등

■ 소재: 퍼, 무스탕, 스웨이드, 니트, 레이스, 데님, 저지, 실크, 면, 울·캐시미어, 코듀로이, 시퀸·글리터, 벨벳, 가죽, 시폰, 합성섬유, 스판덱스, 트위드, 자카드, 메시, 플리스, 네오프렌 등

■ 기장: 상의(크롭·노멀·롱), 하의(미니·니렝스·미디·발목·맥시), 아우터(크롭·노멀·하프·롱·맥시), 원피스(미니·니렝스·미디·발목·맥시)

■ 소매기장: 민소매, 반팔, 캡, 7부소매, 긴팔

■ 넥라인: 라운드넥, 유넥, 브이넥, 홀터넥, 오프숄더, 원숄더, 스퀘어넥, 노카라, 후드, 터틀넥, 보트넥, 스위트하트 등

■ 칼라(옷깃): 셔츠칼라, 피터팬칼라, 보우칼라, 너치드칼라, 세일러칼라, 차이나칼라, 숄칼라, 테일러드칼라, 폴로칼라, 밴드칼라 등

■ 핏: 노멀, 스키니, 루즈, 와이드, 오버사이즈, 타이트, 벨보텀(하의)

■ 실루엣: 페플럼, 머메이드, 비대칭, 벨보텀·플레어, 부츠컷, 펜슬, 테이퍼드, A라인, 스트레이트, H라인, X라인, T/Y라인, O벌크 등

■ JSON 라벨링 필드(영역별): 스타일·서브스타일, 카테고리, 색상, 디테일[], 소재[], 프린트[], 핏, 기장, 소매기장, 넥라인, 옷깃(칼라)

■ 팀 수동 라벨(추가): 패턴 위치, 패턴 크기, 끝단 마감

■ 기타 메타(추측 금지): 섬유조성, 신축성, 비침(공식 라벨)

제약:

1. dense_caption (보완 물리 디테일 — 고밀도 명사구):
- 이미 라벨된 항목은 dense_caption에 넣지 마십시오.
- 의류의 보완 시각 정보만 건조한 명사구로 쉼표로 나열하십시오. 완성 문장 금지.
- long-tail 검색용: 패턴 방향·선색, 시각 질감·워싱, 개폐·하드웨어·스티치, 실루엣, negative_visible 등
- 패턴 위치·패턴 크기·끝단 마감·신체 부위 위치 서술 금지.

2. mood_and_tpo: 주관·TPO 키워드 3~5개. 공식 스타일명 그대로 복사 금지.

출력 형식: JSON만 반환. 마크다운 금지.

{
  "dense_caption": "보완 디테일 명사구 나열",
  "mood_and_tpo": ["태그1", "태그2", "태그3"]
}"""

if API_KEY:
    genai.configure(api_key=API_KEY)
    _model = genai.GenerativeModel(MODEL)
else:
    _model = None


def build_region_prompt(garment_category: str) -> str:
    return (
        SYSTEM_PROMPT
        + f"\n\n이 영역의 폴리곤 대분류 라벨: {garment_category}. "
        "세부 카테고리명은 출력하지 말고, 해당 영역 픽셀의 보완 디테일만 기술하십시오."
    )


def _fetch_image_bytes(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def image_key_from_url(url: str) -> str:
    """공개 URL → R2 object key (예: image/modern/1106011.jpg)."""
    return urlparse(url).path.lstrip("/")


def load_image_and_label(
    *,
    image_url: str | None = None,
    image_key: str | None = None,
    label_key: str | None = None,
    s3_client=None,
    config: R2Config | None = None,
) -> tuple[Image.Image, dict[str, Any], str, str]:
    """원본 이미지 + K-Fashion 라벨 JSON 로드."""
    if config is None:
        config = R2Config.from_env()
    if s3_client is None:
        s3_client = create_r2_client(config)

    if image_url:
        ikey = image_key_from_url(image_url)
    elif image_key:
        ikey = image_key if "/" in image_key else f"{config.image_prefix}{image_key}"
    else:
        raise ValueError("image_url 또는 image_key 필요")

    lkey = label_key or resolve_label_key_from_image(ikey, config)
    label_raw = load_json_from_r2(s3_client, config, lkey)

    try:
        resolved_ikey = resolve_image_key_from_label(
            s3_client, config, lkey, label_raw
        )
        image = load_image_from_r2(s3_client, config, resolved_ikey)
        ikey = resolved_ikey
    except FileNotFoundError:
        if not image_url:
            raise
        image = Image.open(BytesIO(_fetch_image_bytes(image_url))).convert("RGB")

    return image, label_raw, ikey, lkey


def iter_masked_regions(
    image: Image.Image,
    label_raw: dict[str, Any],
    *,
    crop: bool = True,
    categories: list[str] | None = None,
):
    """폴리곤별 마스킹 이미지 생성 (노트북과 동일 로직)."""
    polygon_root = (
        label_raw.get("데이터셋 정보", {})
        .get("데이터셋 상세설명", {})
        .get("폴리곤좌표", {})
    )
    target = categories or GARMENT_CATEGORIES

    for category in target:
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
            yield category, polygon_index, masked


def generate_caption_image(
    image: Image.Image,
    *,
    garment_category: str | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """마스킹된 PIL 이미지 1장 → 캡션 JSON."""
    if _model is None:
        raise RuntimeError("API_KEY가 설정되지 않았습니다.")

    prompt = (
        build_region_prompt(garment_category)
        if garment_category
        else SYSTEM_PROMPT
    )

    for attempt in range(retries):
        try:
            response = _model.generate_content(
                [prompt, image],
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 800,
                },
            )
            return parse_json_response(response.text or "")
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                raise exc
    return {}


def generate_element_captions(
    *,
    image_url: str | None = None,
    image_key: str | None = None,
    label_key: str | None = None,
    crop: bool = True,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """
    한 장의 착용샷 → 폴리곤 요소별 마스킹 캡션 목록.

    반환:
      image_key, label_key, element_captions[{garment_category, polygon_index, dense_caption, mood_and_tpo, error?}]
    """
    config = R2Config.from_env()
    s3 = create_r2_client(config)
    image, label_raw, ikey, lkey = load_image_and_label(
        image_url=image_url,
        image_key=image_key,
        label_key=label_key,
        s3_client=s3,
        config=config,
    )

    elements: list[dict[str, Any]] = []
    for category, polygon_index, masked_image in iter_masked_regions(
        image, label_raw, crop=crop, categories=categories
    ):
        entry: dict[str, Any] = {
            "garment_category": category,
            "polygon_index": polygon_index,
            "masked": True,
        }
        try:
            cap = generate_caption_image(masked_image, garment_category=category)
            entry["dense_caption"] = cap.get("dense_caption", "")
            entry["mood_and_tpo"] = cap.get("mood_and_tpo", [])
        except Exception as exc:
            entry["dense_caption"] = ""
            entry["mood_and_tpo"] = []
            entry["error"] = str(exc)
        elements.append(entry)
        time.sleep(0.3)

    return {
        "image_key": ikey,
        "label_key": lkey,
        "image_file": Path(ikey).name,
        "element_count": len(elements),
        "element_captions": elements,
    }


def batch_generate(
    tasks_path: str,
    out_path: str,
    *,
    limit: int | None = None,
    crop: bool = True,
):
    """tasks JSON 각 행: image URL → 요소별 element_captions 추가."""
    with open(tasks_path, encoding="utf-8") as f:
        tasks = json.load(f)

    if limit:
        tasks = tasks[:limit]

    results = []
    for i, task in enumerate(tasks):
        url = task["data"]["image"]
        name = url.split("/")[-1].split("?")[0]
        print(f"[{i + 1}/{len(tasks)}] {name}")
        try:
            packed = generate_element_captions(image_url=url, crop=crop)
            task["data"]["element_captions"] = packed["element_captions"]
            task["data"]["caption_meta"] = {
                "image_key": packed["image_key"],
                "label_key": packed["label_key"],
                "element_count": packed["element_count"],
            }
            # 하위 호환: 첫 요소를 이미지 레벨 필드에도 복사
            if packed["element_captions"]:
                first = packed["element_captions"][0]
                task["data"]["dense_caption"] = first.get("dense_caption", "")
                task["data"]["mood_and_tpo"] = first.get("mood_and_tpo", [])
            print(f"    OK — {packed['element_count']} region(s)")
        except Exception as exc:
            task["data"]["element_captions"] = []
            task["data"]["dense_caption"] = ""
            task["data"]["mood_and_tpo"] = []
            task["data"]["caption_meta"] = {"error": str(exc)}
            print(f"    ERR — {exc}")

        results.append(task)
        time.sleep(0.5)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n→ {out_path} 저장 완료 ({len(results)}개 이미지)")


def main() -> None:
    parser = argparse.ArgumentParser(description="폴리곤 마스킹 요소별 Gemini 캡셔닝")
    parser.add_argument("--url", type=str, help="단일 이미지 공개 URL")
    parser.add_argument("--batch", type=str, help="tasks JSON 경로")
    parser.add_argument("--out", type=str, default="tasks_captioned.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-crop", action="store_true", help="bbox crop 비활성화")
    args = parser.parse_args()

    crop = not args.no_crop

    if args.url:
        result = generate_element_captions(image_url=args.url, crop=crop)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.batch:
        batch_generate(args.batch, args.out, limit=args.limit, crop=crop)
        return

    # 기본 테스트
    test_url = "https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev/image/modern/1106011.jpg"
    result = generate_element_captions(image_url=test_url, crop=crop)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
