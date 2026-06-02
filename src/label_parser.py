"""
label_parser.py — R2에서 받은 라벨 JSON 파싱 및 인덱스 캐시 관리
"""
import json
import pandas as pd
from pathlib import Path
from src.config import CLOTHING_TYPES


# ─────────────────────────────────────────────────────────────────────────────
# JSON 인덱스 (file_id → {items, iw, ih})
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_file(path: Path) -> tuple[int, dict]:
    """JSON 파일 하나를 파싱해 (file_id, entry) 반환"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    img_info    = data["이미지 정보"]
    detail_info = data["데이터셋 정보"]["데이터셋 상세설명"]
    labeling    = detail_info["라벨링"]
    rect_coords = detail_info["렉트좌표"]
    file_id     = data["데이터셋 정보"]["파일 번호"]
    iw          = img_info["이미지 너비"]
    ih          = img_info["이미지 높이"]

    items = []
    for ct in CLOTHING_TYPES:
        for i, item in enumerate(labeling.get(ct, [{}])):
            if not item or not any(item.values()):
                continue
            rects = rect_coords.get(ct, [{}])
            rect  = rects[i] if i < len(rects) else {}
            bbox  = None
            if rect.get("X좌표") is not None:
                bbox = [rect["X좌표"], rect["Y좌표"], rect["가로"], rect["세로"]]

            category  = item.get("카테고리") or ct
            color_str = item.get("색상") or "-"
            if item.get("서브색상"):
                color_str += f" / {item['서브색상']}"

            items.append({
                "clothing_type": ct,
                "category":      category,
                "color":         item.get("색상"),
                "sub_color":     item.get("서브색상"),
                "fit":           item.get("핏"),
                "length":        item.get("기장"),
                "materials":     item.get("소재", []),
                "details":       item.get("디테일", []),
                "prints":        item.get("프린트", []),
                "item_info": "\n".join([
                    f"타입: {ct} / {category}",
                    f"컬러: {color_str}",
                    f"핏: {item.get('핏') or '-'}",
                    f"기장: {item.get('기장') or '-'}",
                    f"소재: {', '.join(item.get('소재', [])) or '-'}",
                    f"디테일: {', '.join(item.get('디테일', [])) or '-'}",
                    f"프린트: {', '.join(item.get('프린트', [])) or '-'}",
                ]),
                "bbox": bbox,
            })

    entry = {"items": items, "iw": iw, "ih": ih}
    return file_id, entry


def build_json_index(labels_dir: Path, cache_path: Path) -> dict:
    """
    labels_dir 하위 모든 JSON을 파싱해 {file_id: {items, iw, ih}} 반환.
    cache_path가 존재하면 캐시를 로드해 재파싱을 건너뜀.
    """
    if cache_path.exists():
        print("캐시 로드 중...")
        with open(cache_path, encoding="utf-8") as f:
            index = {int(k): v for k, v in json.load(f).items()}
        print(f"캐시 로드 완료: {len(index)}개")
        return index

    print("JSON 인덱싱 중... (최초 1회, 이후 캐시 사용)")
    index = {}
    for path in labels_dir.glob("**/*.json"):
        try:
            file_id, entry = _parse_json_file(path)
            index[file_id] = entry
        except Exception as e:
            print(f"  ⚠ 파싱 실패 {path.name}: {e}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"인덱싱 완료 및 캐시 저장: {len(index)}개")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# 메모리 JSON → 행 (R2 스트리밍 파이프라인용)
# ─────────────────────────────────────────────────────────────────────────────

def rows_from_label_json(data: dict, *, style: str = "") -> list[dict]:
    """R2에서 받은 라벨 dict 한 건을 load_labels()와 동일한 row dict 리스트로 변환."""
    img_info = data["이미지 정보"]
    detail_info = data["데이터셋 정보"]["데이터셋 상세설명"]
    labeling = detail_info["라벨링"]
    rect_coords = detail_info["렉트좌표"]
    file_id = data["데이터셋 정보"]["파일 번호"]
    style_name = style or labeling.get("스타일", [{}])[0].get("스타일", "")
    iw = img_info["이미지 너비"]
    ih = img_info["이미지 높이"]

    rows: list[dict] = []
    for ct in CLOTHING_TYPES:
        for i, item in enumerate(labeling.get(ct, [{}])):
            if not item.get("카테고리"):
                continue
            rects = rect_coords.get(ct, [{}])
            rect = rects[i] if i < len(rects) else {}
            bbox = None
            if rect.get("X좌표") is not None:
                bbox = [rect["X좌표"], rect["Y좌표"], rect["가로"], rect["세로"]]

            rows.append({
                "file_id": file_id,
                "style": style_name,
                "clothing_type": ct,
                "category": item.get("카테고리"),
                "color": item.get("색상"),
                "sub_color": item.get("서브색상"),
                "fit": item.get("핏"),
                "length": item.get("기장"),
                "materials": item.get("소재", []),
                "details": item.get("디테일", []),
                "prints": item.get("프린트", []),
                "bbox": bbox,
                "image_width": iw,
                "image_height": ih,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame 로더 (create_tasks.py 전용 파이프라인용)
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(labels_dir: Path) -> pd.DataFrame:
    """
    labels_dir 하위 JSON을 모두 읽어 의류별로 행을 만든 DataFrame 반환.
    (style 컬럼 포함 — create_tasks.py 샘플링 파이프라인에서 사용)
    """
    rows = []
    for path in labels_dir.glob("**/*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            img_info    = data["이미지 정보"]
            detail_info = data["데이터셋 정보"]["데이터셋 상세설명"]
            labeling    = detail_info["라벨링"]
            rect_coords = detail_info["렉트좌표"]
            file_id     = data["데이터셋 정보"]["파일 번호"]
            style       = labeling.get("스타일", [{}])[0].get("스타일", "")
            iw          = img_info["이미지 너비"]
            ih          = img_info["이미지 높이"]

            for ct in CLOTHING_TYPES:
                for i, item in enumerate(labeling.get(ct, [{}])):
                    if not item.get("카테고리"):
                        continue
                    rects = rect_coords.get(ct, [{}])
                    rect  = rects[i] if i < len(rects) else {}
                    bbox  = None
                    if rect.get("X좌표") is not None:
                        bbox = [rect["X좌표"], rect["Y좌표"], rect["가로"], rect["세로"]]

                    rows.append({
                        "file_id":       file_id,
                        "style":         style,
                        "clothing_type": ct,
                        "category":      item.get("카테고리"),
                        "color":         item.get("색상"),
                        "sub_color":     item.get("서브색상"),
                        "fit":           item.get("핏"),
                        "length":        item.get("기장"),
                        "materials":     item.get("소재", []),
                        "details":       item.get("디테일", []),
                        "prints":        item.get("프린트", []),
                        "bbox":          bbox,
                        "image_width":   iw,
                        "image_height":  ih,
                    })
        except Exception as e:
            print(f"  ⚠ 파싱 실패 {path.name}: {e}")

    return pd.DataFrame(rows)
