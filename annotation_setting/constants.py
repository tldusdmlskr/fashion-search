"""K-fashion 24 스타일 폴더명 (R2 image/ · labeling/ 공통)."""

STYLES = [
    "avant_garde",
    "classic",
    "country",
    "etc",
    "feminine",
    "genderless",
    "hiphop",
    "hippie",
    "kitsch",
    "mannish",
    "military",
    "modern",
    "oriental",
    "preppy",
    "punk",
    "resort",
    "retro",
    "romantic",
    "sexy",
    "sophisticated",
    "sporty",
    "street",
    "tomboy",
    "western",
]

CLOTHING_TYPES = ("상의", "하의", "아우터", "원피스")

PRINT_ATTR = "프린트"
PRINT_VALUE = "스트라이프"

DEFAULT_PER_STYLE = 40
TARGET_TOTAL = len(STYLES) * DEFAULT_PER_STYLE  # 960
