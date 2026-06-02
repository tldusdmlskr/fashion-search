"""
sampler.py — 다양성 기반 샘플링
"""
import pandas as pd
from src.config import CLOTHING_TYPES


def sample_diverse(df: pd.DataFrame, n_total: int = 1500, color_cap: int = 50) -> pd.DataFrame:
    """
    clothing_type → category → color 계층으로 균등 샘플링.

    Args:
        df:        load_labels()가 반환한 DataFrame
        n_total:   목표 총 샘플 수
        color_cap: 카테고리 내 특정 색상의 최대 포함 수

    Returns:
        셔플된 샘플 DataFrame
    """
    samples = []
    present_types = [ct for ct in CLOTHING_TYPES if ct in df["clothing_type"].values]
    n_per_type = n_total // len(present_types)

    for ct in present_types:
        type_df    = df[df["clothing_type"] == ct]
        n_per_cat  = max(1, n_per_type // type_df["category"].nunique())

        for cat in type_df["category"].unique():
            cat_df = type_df[type_df["category"] == cat]
            capped = (
                cat_df.groupby("color", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), color_cap), random_state=42))
            )
            samples.append(capped.sample(min(len(capped), n_per_cat), random_state=42))

    result = pd.concat(samples).drop_duplicates(subset=["file_id", "clothing_type"])
    return result.sample(frac=1, random_state=42).reset_index(drop=True)
