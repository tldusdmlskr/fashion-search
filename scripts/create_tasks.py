"""
create_tasks.py — 전체 labels JSON 로드 → 다양성 샘플링 → task 생성 → output/tasks.json
실행: python scripts/create_tasks.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.label_parser import load_labels
from src.sampler      import sample_diverse
from src.task_builder import make_tasks_from_df

LABELS_DIR = Path(__file__).parent.parent / "data" / "labels"
OUT_PATH   = Path(__file__).parent.parent / "output" / "tasks.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

N_TOTAL   = 1500
COLOR_CAP = 50

print("JSON 파일 로딩 중...")
df = load_labels(LABELS_DIR)
print(f"전체 라벨 수: {len(df)}")
print(f"\n[clothing_type 분포]\n{df['clothing_type'].value_counts()}")

sampled = sample_diverse(df, n_total=N_TOTAL, color_cap=COLOR_CAP)
print(f"\n샘플링 결과: {len(sampled)}개")
print(f"\n[샘플 clothing_type 분포]\n{sampled['clothing_type'].value_counts()}")

tasks = make_tasks_from_df(sampled)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"\n→ {OUT_PATH.name} 저장 완료 ({len(tasks)}개 task)")
print(json.dumps(tasks[0], ensure_ascii=False, indent=2))
