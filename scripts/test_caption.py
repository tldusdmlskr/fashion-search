"""
test_caption.py — VLM 캡션 품질 평가 스크립트
test_data/ 의 JPG + JSON 쌍으로 폴리곤 마스킹 후 Gemini 캡션 생성.
결과를 test_data/caption_test_results.jsonl 에 저장하고 콘솔에 요약 출력.

실행: python scripts/test_caption.py
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.caption import generate_caption_from_image, build_dense_caption, build_flat_tags, CaptionOutput
from src.masking import load_masked_items

TEST_DIR = Path(__file__).parent.parent / "test_data"
OUT_PATH = TEST_DIR / "caption_test_results.jsonl"

_CAT_MAP = {"상의": "상의", "하의": "하의", "아우터": "아우터", "원피스": "원피스"}
_EXISTING_KEYS = ["카테고리", "색상", "서브색상", "소재", "핏", "기장",
                  "소매기장", "넥라인", "프린트", "디테일"]


def _extract_existing(json_path: Path, category_key: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    labels = (data.get("데이터셋 정보", {})
                  .get("데이터셋 상세설명", {})
                  .get("라벨링", {}))
    items = labels.get(category_key, [{}])
    item = items[0] if items else {}
    return {k: item.get(k) for k in _EXISTING_KEYS if item.get(k)}


def main():
    jpg_files = sorted(TEST_DIR.glob("*.jpg"))
    print(f"\n테스트 이미지: {len(jpg_files)}개\n{'='*70}")

    results = []
    for jpg_path in jpg_files:
        json_path = jpg_path.with_suffix(".json")
        if not json_path.exists():
            continue

        masked_list = load_masked_items(json_path, jpg_path)
        if not masked_list:
            print(f"  {jpg_path.name}: 폴리곤 없음, 스킵\n")
            continue

        for cat, idx, masked_img in masked_list:
            label = f"{jpg_path.stem} [{cat}#{idx}]"
            print(f"{'─'*70}")
            print(f"  {label}")

            existing = _extract_existing(json_path, _CAT_MAP.get(cat, cat))
            ex_summary = (f"카테고리={existing.get('카테고리','-')} "
                          f"색상={existing.get('색상','-')} "
                          f"소재={existing.get('소재','-')} "
                          f"핏={existing.get('핏','-')} "
                          f"프린트={existing.get('프린트','-')}")
            print(f"  [기존] {ex_summary}")

            try:
                t0 = time.time()
                vlm: CaptionOutput = generate_caption_from_image(masked_img, retries=5)
                elapsed = time.time() - t0

                dense = build_dense_caption(vlm, existing or None)
                tags  = build_flat_tags(vlm, existing or None)

                print(f"  [VLM {elapsed:.1f}s]")
                print(f"    category     : {vlm.get('category','')}")
                print(f"    micro_details: {vlm.get('micro_details',[])}")
                print(f"    mood_and_tpo : {vlm.get('mood_and_tpo',[])}")
                print(f"    dense_caption: {dense}")
                print(f"    flat_tags    : {tags}")

                row = {
                    "file": jpg_path.stem,
                    "category_label": cat,
                    "existing": existing,
                    "vlm": vlm,
                    "dense_caption": dense,
                    "flat_tags": tags,
                    "elapsed_s": round(elapsed, 2),
                    "ok": True,
                }
            except Exception as e:
                err_str = str(e)
                if "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in err_str:
                    print(f"  SKIP: 일일 무료 한도 소진 — paid tier 필요")
                else:
                    print(f"  ERROR: {e}")
                row = {"file": jpg_path.stem, "category_label": cat,
                       "error": str(e), "ok": False}

            results.append(row)
            print()

    # 저장
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok  = sum(1 for r in results if r.get("ok"))
    err = len(results) - ok
    print(f"{'='*70}")
    print(f"완료: {ok}건 성공 / {err}건 실패  →  {OUT_PATH}")

    # 품질 요약 출력
    if ok:
        print(f"\n{'─'*70}")
        print("품질 요약:")
        for r in results:
            if not r.get("ok"):
                continue
            vlm = r["vlm"]
            issues = []
            tpo = vlm.get("mood_and_tpo", [])
            if "데일리룩" in tpo and "캐주얼" in tpo:
                issues.append("데일리룩+캐주얼 중복")
            bad_adj = [t for t in tpo if t in ("편안한", "시원한", "아름다운", "여유로운", "활동적인")]
            if bad_adj:
                issues.append(f"형용사: {bad_adj}")
            allowed_tpo = {"하객룩","오피스룩","데이트룩","피크닉룩","바캉스룩","페스티벌룩",
                           "등산룩","골프룩","리조트룩","웨딩게스트","미니멀룩","고프코어",
                           "Y2K","뉴트로","페미닌","클래식","스트리트","이지웨어","데일리룩","캐주얼"}
            unknown = [t for t in tpo if t not in allowed_tpo]
            if unknown:
                issues.append(f"태그풀 외: {unknown}")
            status = "OK" if not issues else "NG: " + " / ".join(issues)
            print(f"  [{r['file']} {r['category_label']}] {status}")


if __name__ == "__main__":
    main()
