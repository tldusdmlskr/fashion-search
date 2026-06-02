from urllib.parse import quote, unquote
from src.config import R2_PUBLIC_URL, R2_BUCKET_PREFIX

STYLE_MAP = {
    "아방가르드":       "avant_garde",
    "클래식":          "classic",
    "컨트리":          "country",
    "기타":            "etc",
    "페미닌":          "feminine",
    "젠더리스":        "genderless",
    "힙합":            "hiphop",
    "히피":            "hippie",
    "키치":            "kitsch",
    "매니시":          "mannish",
    "밀리터리":        "military",
    "모던":            "modern",
    "오리엔탈":        "oriental",
    "프레피":          "preppy",
    "펑크":            "punk",
    "리조트":          "resort",
    "레트로":          "retro",
    "로맨틱":          "romantic",
    "섹시":            "sexy",
    "소피스트케이티드": "sophisticated",
    "스포티":          "sporty",
    "스트리트":        "street",
    "톰보이":          "tomboy",
    "웨스턴":          "western",
}


def build_image_url(style: str, file_id: int) -> str:
    """스타일(영문) + 파일번호 → R2 이미지 URL"""
    folder = STYLE_MAP.get(style, style)  # 한글이면 변환, 이미 영문이면 그대로
    return f"{R2_PUBLIC_URL}/{R2_BUCKET_PREFIX}/{folder}/{file_id}.jpg"


def fix_url(url: str) -> str:
    """한글 폴더명이 포함된 URL을 영문으로 수정"""
    decoded = unquote(url)
    for ko, en in STYLE_MAP.items():
        if f"/image/{ko}/" in decoded:
            return decoded.replace(f"/image/{ko}/", f"/image/{en}/")
    return decoded
