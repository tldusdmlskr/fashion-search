"""
masking.py — 폴리곤 마스킹 유틸리티
JSON 라벨의 폴리곤 좌표로 의류 영역만 남기고 배경을 순백색으로 처리.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLOTHING_CATEGORIES = ["상의", "하의", "아우터", "원피스"]


def extract_polygon_points(polygon_data: dict) -> list[tuple[int, int]]:
    """JSON 폴리곤 dict에서 (x, y) 좌표 목록 추출."""
    points = []
    for i in range(1, 100):
        x = polygon_data.get(f"X좌표{i}")
        y = polygon_data.get(f"Y좌표{i}")
        if x is None or y is None:
            break
        points.append((int(x), int(y)))
    return points


def apply_polygon_mask(
    image: Image.Image,
    points: list[tuple[int, int]],
    *,
    crop: bool = True,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    폴리곤 영역만 남기고 배경을 bg_color로 채운 이미지 반환.

    Args:
        image:    원본 PIL Image (RGB)
        points:   폴리곤 꼭짓점 좌표 목록
        crop:     True면 의류 bbox로 추가 크롭
        bg_color: 배경 색상 (기본 순백색)
    """
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=1)
    mask_np = np.array(mask)

    img_np   = np.array(image)
    result   = np.full_like(img_np, bg_color)
    result[mask_np == 1] = img_np[mask_np == 1]

    if crop:
        ys, xs = np.where(mask_np == 1)
        if len(xs) == 0:
            return Image.fromarray(result)
        result = result[ys.min():ys.max(), xs.min():xs.max()]

    return Image.fromarray(result)


def load_masked_items(
    json_path: Path,
    image_path: Path,
    *,
    categories: list[str] | None = None,
    crop: bool = True,
) -> list[tuple[str, int, Image.Image]]:
    """
    JSON + 이미지 파일에서 카테고리별 마스킹 이미지 목록 반환.

    Returns:
        [(category, polygon_index, masked_image), ...]
    """
    import json

    target = categories or CLOTHING_CATEGORIES

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    image = Image.open(image_path).convert("RGB")
    polygon_root = (
        data.get("데이터셋 정보", {})
        .get("데이터셋 상세설명", {})
        .get("폴리곤좌표", {})
    )

    results = []
    for cat in target:
        for idx, poly_data in enumerate(polygon_root.get(cat, [])):
            if not poly_data:
                continue
            points = extract_polygon_points(poly_data)
            if len(points) < 3:
                continue
            masked = apply_polygon_mask(image, points, crop=crop)
            results.append((cat, idx, masked))

    return results
