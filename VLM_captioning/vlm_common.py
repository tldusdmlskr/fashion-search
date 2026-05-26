"""Cloudflare R2 이미지 로드 및 K-fashion VLM 캡셔닝 공통 유틸."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import boto3
import numpy as np
from botocore.config import Config
from PIL import Image, ImageDraw

_VLM_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(_VLM_DIR / ".env")
except ImportError:
    pass

OUTPUT_DIR = _VLM_DIR / "output"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GARMENT_CATEGORIES = ["상의", "하의", "아우터", "원피스"]
MAX_POLYGON_POINTS = 64

# AI Hub GT + 수동 라벨링으로 이미 확보된 항목 — VLM 출력 금지
EXCLUDED_LABELS_KO = """
- AI Hub / 분류 체계: 세부 카테고리, 공식 컬러, 프린트 종류, 소재, 기장, 소매기장, 넥라인, 칼라, 핏, 스타일·무드
- AI Hub 원본 메타: 섬유조성, 신축성, 비침
- 팀 수동 라벨: 패턴 위치, 패턴 크기, 끝단 마감
""".strip()

FASHION_DETAIL_PROMPT = f"""
You analyze a K-fashion garment in a wear-shot image. Describe ONLY complementary visual details
that are NOT already stored as structured labels. Ignore background, face, hair, skin, and body shape.

[DO NOT output or guess — already labeled elsewhere]
{EXCLUDED_LABELS_KO}

[DO extract — complementary attributes visible in the image]

Return JSON only with this exact schema:

{{
  "complementary_attributes": {{
    "pattern": {{
      "direction": "",
      "line_or_accent_colors": [],
      "base_color_visible": "",
      "density": ""
    }},
    "surface_finish": {{
      "sheen": "",
      "texture_visible": "",
      "wash_or_finish_look": "",
      "drape_stiffness": "",
      "sheer_visible": ""
    }},
    "construction": {{
      "closure": "",
      "hardware": "",
      "pockets": "",
      "stitching_trim": "",
      "asymmetry_or_cutout": "",
      "layering_visible": ""
    }},
    "graphic": {{
      "present": false,
      "type": "",
      "description": "",
      "location": ""
    }},
    "silhouette_proportion": {{
      "overall_shape": "",
      "rise_or_waist": "",
      "shoulder_sleeve_volume": "",
      "taper_or_flare": ""
    }},
    "detail_color": "",
    "negative_visible": []
  }},
  "dense_caption_ko": "",
  "dense_caption_en": "",
  "search_phrases_ko": []
}}

Field rules:

1. pattern.direction: one of ["vertical", "horizontal", "diagonal", "mixed", "none", "unknown"]
   — Do NOT describe pattern position on garment or pattern scale/size (already manually labeled).

2. pattern.density: visual only — e.g. "sparse", "dense", "unknown" (NOT pattern size label).

3. surface_finish.sheer_visible: photo-visible transparency only — do NOT output official sheerness.

4. construction: describe closures, hardware, pockets, contrast stitching, asymmetry, layering.
   — Do NOT describe hem finishing / raw hem / cuff type (끝단 마감 — manually labeled).

5. graphic: if absent, set present=false and other graphic fields to "none".

6. detail_color: color of buttons, zippers, stripes, topstitch, small decorations; or "none".

7. negative_visible: explicitly absent elements visible from image
   (e.g. "no logo", "no hood", "no visible lining").

8. dense_caption_ko / dense_caption_en: 2–3 sentences using long-tail search phrasing
   (material feel, pattern direction, hardware, silhouette — exclude items in DO NOT list).

9. search_phrases_ko: 5–12 short Korean keywords/phrases for sparse retrieval; no duplicates of excluded labels.

10. Use "none", "unknown", or false when not visible. Use [] for empty lists.

Return JSON only. No markdown fences.
""".strip()


def build_prompt(item: CaptionItem | None = None) -> str:
    """마스킹 항목이면 라벨 카테고리 힌트를 프롬프트에 추가."""
    extra = ""
    if item and item.masked and item.garment_category:
        extra = (
            f"\n\nThe isolated garment region is labeled: {item.garment_category}. "
            "Analyze only pixels inside this garment mask. "
            "Do not infer category/subcategory names — focus on complementary attributes only."
        )
    return FASHION_DETAIL_PROMPT + extra


GEMINI_ELEMENT_PROMPT = """
역할:
당신은 한국 이커머스 패션 플랫폼의 전문 카탈로그 데이터 구축 모델입니다.

과제:
입력 이미지는 착용샷에서 폴리곤으로 분리·배경 제거된 단일 의류 영역입니다.
(1) 보완 물리 디테일 캡션과 (2) 감성·TPO 태그를 JSON으로 추출하십시오.
배경·얼굴·다른 의류 영역은 이미 제거되었습니다. 보이는 의류 픽셀만 분석하십시오.

[출력 금지 — K-Fashion 데이터셋·JSON에 이미 존재하는 속성. 추측·서술·dense_caption 반복 금지]
(출처: K-Fashion 이미지 데이터 구축 가이드라인 — 11개 상위 속성, 스타일 23종, 세부속성 186종)

■ 스타일(다중 레이블): 클래식, 프레피, 매니시, 톰보이, 로맨틱, 섹시, 힙합, 스트리트, 리조트, 미니멀 등 23종
■ 카테고리·아이템: 상의/하의/아우터/원피스 세부 품목 전체
■ 컬러, 디테일, 프린트, 소재, 기장, 소매기장, 넥라인, 칼라, 핏, 실루엣
■ JSON 필드: 스타일, 색상, 디테일[], 소재[], 프린트[], 핏, 기장, 소매기장, 넥라인, 옷깃
■ 팀 수동 라벨: 패턴 위치, 패턴 크기, 끝단 마감
■ 기타 메타(추측 금지): 섬유조성, 신축성, 비침

제약:
1. dense_caption: 보완 시각 정보만 건조한 명사구(쉼표 구분). 완성 문장 금지.
   패턴 방향·선색, 시각 질감·워싱, 개폐·하드웨어·스티치, 실루엣, negative_visible 등.
   패턴 위치·패턴 크기·끝단 마감·신체 부위 위치 서술 금지.
2. mood_and_tpo: 주관·TPO 키워드 3~5개. 공식 스타일명 그대로 복사 금지.

출력 형식: JSON만 반환. 마크다운 금지.
{
  "dense_caption": "보완 디테일 명사구 나열",
  "mood_and_tpo": ["태그1", "태그2", "태그3"]
}
""".strip()


def build_gemini_element_prompt(item: CaptionItem | None = None) -> str:
    """Gemini 요소별 캡션용 프롬프트 (폴리곤 마스킹 영역)."""
    if item and item.masked and item.garment_category:
        return (
            GEMINI_ELEMENT_PROMPT
            + f"\n\n이 영역의 폴리곤 대분류 라벨: {item.garment_category}. "
            "세부 카테고리명은 출력하지 말고, 해당 영역 픽셀의 보완 디테일만 기술하십시오."
        )
    return GEMINI_ELEMENT_PROMPT


@dataclass
class R2Config:
    account_id: str
    access_key: str
    secret_key: str
    bucket: str
    image_prefix: str
    label_prefix: str

    @classmethod
    def from_env(cls) -> "R2Config":
        account_id = os.environ.get("R2_ACCOUNT_ID", "")
        access_key = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY", "")
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_KEY", "")
        bucket = os.environ.get("R2_BUCKET_NAME", "project2")
        image_prefix = os.environ.get("R2_IMAGE_PREFIX", "image/test/")
        label_prefix = os.environ.get("R2_LABEL_PREFIX", "labeling/test/")

        missing = [
            name
            for name, value in [
                ("R2_ACCOUNT_ID", account_id),
                ("R2_ACCESS_KEY_ID", access_key),
                ("R2_SECRET_ACCESS_KEY", secret_key),
            ]
            if not value
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        if image_prefix and not image_prefix.endswith("/"):
            image_prefix += "/"
        if label_prefix and not label_prefix.endswith("/"):
            label_prefix += "/"

        return cls(
            account_id=account_id,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            image_prefix=image_prefix,
            label_prefix=label_prefix,
        )


@dataclass
class CaptionItem:
    image_key: str
    image: Image.Image
    masked: bool = False
    label_key: str | None = None
    garment_category: str | None = None
    polygon_index: int | None = None

    @property
    def display_name(self) -> str:
        if self.masked and self.garment_category is not None:
            return (
                f"{self.image_key} [{self.garment_category}"
                f"#{self.polygon_index}] (masked)"
            )
        return self.image_key


def create_r2_client(config: R2Config):
    return boto3.client(
        service_name="s3",
        endpoint_url=f"https://{config.account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        config=Config(signature_version="s3v4"),
    )


def iter_image_keys(
    s3_client,
    config: R2Config,
    *,
    limit: int | None = None,
    keys: list[str] | None = None,
) -> Iterator[str]:
    """R2 prefix 아래 이미지 object key 목록."""
    if keys:
        for key in keys:
            normalized = key if "/" in key else f"{config.image_prefix}{key}"
            yield normalized
        return

    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=config.bucket, Prefix=config.image_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if Path(key).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            yield key
            count += 1
            if limit is not None and count >= limit:
                return


def load_image_from_r2(s3_client, config: R2Config, key: str) -> Image.Image:
    response = s3_client.get_object(Bucket=config.bucket, Key=key)
    return Image.open(BytesIO(response["Body"].read())).convert("RGB")


def load_json_from_r2(s3_client, config: R2Config, key: str) -> dict[str, Any]:
    response = s3_client.get_object(Bucket=config.bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def image_filename_from_key(key: str) -> str:
    return Path(key).name


def normalize_image_key(config: R2Config, key: str) -> str:
    if "/" in key:
        return key
    return f"{config.image_prefix}{key}"


def normalize_label_key(config: R2Config, key: str) -> str:
    name = Path(key).name
    if not name.endswith(".json"):
        name = f"{Path(name).stem}.json"
    if "/" in key:
        return key if key.endswith(".json") else f"{key}.json"
    return f"{config.label_prefix}{name}"


def label_key_for_image(config: R2Config, image_filename: str) -> str:
    stem = Path(image_filename).stem
    return f"{config.label_prefix}{stem}.json"


def image_key_for_filename(config: R2Config, image_filename: str) -> str:
    return normalize_image_key(config, image_filename)


def resolve_label_key_from_image(image_key: str, config: R2Config) -> str:
    """
    이미지 R2 key → 라벨 JSON key.
    예: image/modern/1106011.jpg → labeling/modern/1106011.json
    """
    if image_key.startswith("image/") and "/" in image_key[len("image/") :]:
        subpath = image_key.split("image/", 1)[1]
        return f"labeling/{Path(subpath).with_suffix('.json').as_posix()}"
    return label_key_for_image(config, Path(image_key).name)


def image_key_candidates(
    label_key: str,
    label_raw: dict[str, Any],
    config: R2Config,
) -> list[str]:
    """
    라벨 JSON 1건에 대응할 이미지 R2 key 후보 (우선순위 순).
    1) JSON '이미지 정보.이미지 파일명' (K-Fashion 정본)
    2) labeling/…/파일.json ↔ image/…/파일.jpg 경로 대칭
    3) R2_IMAGE_PREFIX + 파일명
    """
    candidates: list[str] = []
    image_info = label_raw.get("이미지 정보", {})
    filename = (image_info.get("이미지 파일명") or "").strip()

    if filename:
        if "/" in filename:
            candidates.append(filename.lstrip("/"))
        basename = Path(filename).name
        candidates.append(normalize_image_key(config, basename))

    if label_key.startswith("labeling/"):
        subpath = label_key[len("labeling/") :]
        stem_path = Path(subpath).with_suffix("")
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidates.append(f"image/{stem_path.as_posix()}{ext}")

    stem = Path(label_key).stem
    candidates.append(normalize_image_key(config, f"{stem}.jpg"))

    seen: set[str] = set()
    ordered: list[str] = []
    for key in candidates:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def resolve_image_key_from_label(
    s3_client,
    config: R2Config,
    label_key: str,
    label_raw: dict[str, Any],
) -> str:
    """라벨 JSON과 짝이 맞는 이미지 key를 R2에서 확인해 반환."""
    tried: list[str] = []
    last_error: Exception | None = None

    for candidate in image_key_candidates(label_key, label_raw, config):
        tried.append(candidate)
        try:
            s3_client.head_object(Bucket=config.bucket, Key=candidate)
            return candidate
        except Exception as exc:
            last_error = exc

    raise FileNotFoundError(
        f"라벨 {label_key}에 대응하는 이미지를 R2에서 찾지 못했습니다. "
        f"시도한 key: {tried}"
    ) from last_error


def extract_polygon_points(polygon_item: dict[str, Any]) -> list[tuple[int, int]]:
    """K-fashion JSON 폴리곤 항목 → (x, y) 픽셀 좌표."""
    points: list[tuple[int, int]] = []
    for idx in range(1, MAX_POLYGON_POINTS + 1):
        x_key = f"X좌표{idx}"
        y_key = f"Y좌표{idx}"
        if x_key not in polygon_item or y_key not in polygon_item:
            break
        points.append((int(polygon_item[x_key]), int(polygon_item[y_key])))
    return points


def apply_polygon_mask(
    image: Image.Image,
    points: list[tuple[int, int]],
    *,
    crop: bool = True,
) -> Image.Image:
    """
    폴리곤 내부만 원본 픽셀 유지, 외부는 검정.
    crop=True이면 마스킹 결과를 bbox로 잘라 반환 (노트북과 동일).
    """
    if len(points) < 3:
        raise ValueError("polygon must have at least 3 points")

    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    mask_np = np.array(mask, dtype=bool)
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


def iter_label_keys(
    s3_client,
    config: R2Config,
    *,
    limit: int | None = None,
    keys: list[str] | None = None,
) -> Iterator[str]:
    if keys:
        for key in keys:
            yield normalize_label_key(config, key)
        return

    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=config.bucket, Prefix=config.label_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            yield key
            count += 1
            if limit is not None and count >= limit:
                return


def iter_masked_caption_items(
    s3_client,
    config: R2Config,
    *,
    crop: bool = True,
    limit: int | None = None,
    keys: list[str] | None = None,
    categories: list[str] | None = None,
) -> Iterator[CaptionItem]:
    """라벨 JSON 폴리곤별 마스킹 이미지 항목 생성."""
    target_categories = categories or GARMENT_CATEGORIES
    count = 0

    for label_key in iter_label_keys(s3_client, config, keys=keys):
        raw = load_json_from_r2(s3_client, config, label_key)
        image_info = raw.get("이미지 정보", {})
        if not image_info.get("이미지 파일명"):
            print(f"  skip (no image filename in JSON): {label_key}")
            continue

        try:
            image_key = resolve_image_key_from_label(
                s3_client, config, label_key, raw
            )
        except FileNotFoundError as exc:
            print(f"  skip: {exc}")
            continue

        image = load_image_from_r2(s3_client, config, image_key)
        polygon_root = (
            raw.get("데이터셋 정보", {})
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

                masked_image = apply_polygon_mask(image, points, crop=crop)
                yield CaptionItem(
                    image_key=image_key,
                    image=masked_image,
                    masked=True,
                    label_key=label_key,
                    garment_category=category,
                    polygon_index=polygon_index,
                )
                count += 1
                if limit is not None and count >= limit:
                    return


def iter_caption_items(
    s3_client,
    config: R2Config,
    *,
    use_mask: bool = False,
    crop: bool = True,
    limit: int | None = None,
    keys: list[str] | None = None,
    categories: list[str] | None = None,
) -> Iterator[CaptionItem]:
    if use_mask:
        yield from iter_masked_caption_items(
            s3_client,
            config,
            crop=crop,
            limit=limit,
            keys=keys,
            categories=categories,
        )
        return

    for image_key in iter_image_keys(
        s3_client, config, limit=limit, keys=keys
    ):
        image = load_image_from_r2(s3_client, config, image_key)
        yield CaptionItem(image_key=image_key, image=image, masked=False)


def parse_json_response(text: str) -> dict[str, Any]:
    """모델 응답에서 JSON 객체 추출."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def build_result_record(
    *,
    model_name: str,
    image_key: str,
    caption: dict[str, Any] | None,
    raw_response: str,
    error: str | None = None,
    masked: bool = False,
    label_key: str | None = None,
    garment_category: str | None = None,
    polygon_index: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model": model_name,
        "image_key": image_key,
        "image_file": image_filename_from_key(image_key),
        "masked": masked,
        "caption": caption,
        "raw_response": raw_response,
        "error": error,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if masked:
        record["label_key"] = label_key
        record["garment_category"] = garment_category
        record["polygon_index"] = polygon_index
    if caption:
        if "dense_caption" in caption:
            record["dense_caption"] = caption["dense_caption"]
        if "mood_and_tpo" in caption:
            record["mood_and_tpo"] = caption["mood_and_tpo"]
    return record


def save_results_jsonl(records: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def default_output_path(model_slug: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{model_slug}_{timestamp}.jsonl"
