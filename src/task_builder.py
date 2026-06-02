"""
task_builder.py — Label Studio task dict 생성 유틸리티
"""
import pandas as pd
from src.url_utils import build_image_url


def build_item_info(ct: str, item: dict) -> str:
    """의류 항목 하나로부터 item_info 문자열 생성"""
    category  = item.get("카테고리") or ct
    color_str = item.get("색상") or "-"
    if item.get("서브색상"):
        color_str += f" / {item['서브색상']}"

    return "\n".join([
        f"타입: {ct} / {category}",
        f"컬러: {color_str}",
        f"핏: {item.get('핏') or '-'}",
        f"기장: {item.get('기장') or '-'}",
        f"소재: {', '.join(item.get('소재', [])) or '-'}",
        f"디테일: {', '.join(item.get('디테일', [])) or '-'}",
        f"프린트: {', '.join(item.get('프린트', [])) or '-'}",
    ])


def make_prediction(bbox: list | None, iw: int, ih: int) -> list:
    """
    바운딩박스를 Label Studio rectanglelabels prediction 형식으로 변환.
    bbox = [X좌표, Y좌표, 가로, 세로] (픽셀 절댓값)
    """
    if not (bbox and iw and ih):
        return []
    return [{
        "model_version": "metadata_bbox",
        "result": [{
            "from_name": "bbox_display",
            "to_name":   "image",
            "type":      "rectanglelabels",
            "value": {
                "x":      (bbox[0] / iw) * 100,
                "y":      (bbox[1] / ih) * 100,
                "width":  (bbox[2] / iw) * 100,
                "height": (bbox[3] / ih) * 100,
                "rectanglelabels": ["현재 라벨링 대상"],
            }
        }]
    }]


def make_task(
    file_id: int,
    image_url: str,
    item_info: str,
    predictions: list,
    **extra_data,
) -> dict:
    """Label Studio 임포트용 task dict 생성"""
    data = {
        "image":     image_url,
        "item_info": item_info,
        "file_id":   file_id,
        **extra_data,
    }
    return {"data": data, "predictions": predictions}


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame 기반 일괄 task 생성 (create_tasks.py 파이프라인용)
# ─────────────────────────────────────────────────────────────────────────────

def make_tasks_from_df(df: pd.DataFrame) -> list[dict]:
    """
    load_labels() + sample_diverse()로 만든 DataFrame을 task 리스트로 변환.
    style 컬럼을 이용해 R2 이미지 URL을 구성함.
    """
    tasks = []
    for _, row in df.iterrows():
        bbox = row["bbox"]
        iw   = row["image_width"]
        ih   = row["image_height"]

        color_str = row["color"] or "-"
        if row.get("sub_color"):
            color_str += f" / {row['sub_color']}"

        item_info = "\n".join([
            f"타입: {row['clothing_type']} / {row['category']}",
            f"컬러: {color_str}",
            f"핏: {row['fit'] or '-'}",
            f"기장: {row['length'] or '-'}",
            f"소재: {', '.join(row['materials']) if row['materials'] else '-'}",
            f"디테일: {', '.join(row['details']) if row['details'] else '-'}",
            f"프린트: {', '.join(row['prints']) if row['prints'] else '-'}",
        ])

        image_url   = build_image_url(row["style"], int(row["file_id"]))
        predictions = make_prediction(bbox, iw, ih)

        tasks.append(make_task(
            file_id=int(row["file_id"]),
            image_url=image_url,
            item_info=item_info,
            predictions=predictions,
            clothing_type=row["clothing_type"],
            category=row["category"],
        ))

    return tasks
