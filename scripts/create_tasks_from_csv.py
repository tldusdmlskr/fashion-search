"""
create_tasks_from_csv.py — CSV 기반 task 생성 (JSON 인덱스 캐시 사용)
실행: python scripts/create_tasks_from_csv.py [csv_path] [out_path]

인자 없이 실행하면 기본값 사용:
  CSV:  data/sample_1500(2).csv
  출력: output/tasks_final.json
"""
import sys
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.label_parser import build_json_index
from src.task_builder import make_prediction

ROOT       = Path(__file__).parent.parent
LABELS_DIR = ROOT / "data" / "labels"
CACHE_PATH = ROOT / "data" / "json_index_cache.json"
CSV_PATH   = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "sample_1500(2).csv"
OUT_PATH   = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "output" / "tasks_final.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── 1. JSON 인덱스 로드 ───────────────────────────────────────────────────────
json_index = build_json_index(LABELS_DIR, CACHE_PATH)

# ── 2. CSV 읽기 ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print(f"CSV 행 수: {len(df)}")

# ── 3. task 생성 ──────────────────────────────────────────────────────────────
tasks   = []
missing = 0

for _, row in df.iterrows():
    file_id   = int(row["image_id"])
    image_url = row["image_url"]
    pp_pred   = row.get("pp_pred", "")
    ps_pred   = row.get("ps_pred", "")
    trim_pred = row.get("trim_pred", "")

    meta = json_index.get(file_id)
    if meta is None:
        missing  += 1
        item_info = f"file_id: {file_id}"
        predictions = []
    else:
        iw, ih = meta["iw"], meta["ih"]
        if meta["items"]:
            best      = meta["items"][0]
            item_info = best.get("item_info", "")
            bbox      = best.get("bbox")
        else:
            item_info = f"스타일: {row.get('folder', '-')}\n(원본 속성 정보 없음)"
            bbox      = None
        predictions = make_prediction(bbox, iw, ih)

    tasks.append({
        "data": {
            "image":     image_url,
            "item_info": item_info,
            "file_id":   file_id,
            "pp_pred":   str(pp_pred),
            "ps_pred":   str(ps_pred),
            "trim_pred": str(trim_pred),
        },
        "predictions": predictions,
    })

# ── 4. 저장 ───────────────────────────────────────────────────────────────────
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"\n→ {OUT_PATH.name} 저장 완료 ({len(tasks)}개 task)")
if missing:
    print(f"  ⚠ JSON 매칭 실패: {missing}개 (image_id 불일치)")
print(json.dumps(tasks[0], ensure_ascii=False, indent=2))
