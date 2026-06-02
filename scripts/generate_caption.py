"""
generate_caption.py — Gemini로 tasks JSON에 dense caption 추가
실행: python scripts/generate_caption.py [tasks_path] [out_path] [limit]

인자 없이 실행하면 단일 테스트 URL로 확인.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.caption import generate_caption, batch_generate

if len(sys.argv) >= 3:
    tasks_path = sys.argv[1]
    out_path   = sys.argv[2]
    limit      = int(sys.argv[3]) if len(sys.argv) >= 4 else None
    batch_generate(tasks_path, out_path, limit=limit)
else:
    # 단일 테스트
    test_url = "https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev/image/modern/1106011.jpg"
    result   = generate_caption(test_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
