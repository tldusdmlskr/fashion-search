"""
sample_1500.csv 분포 확인
--------------------------
- 버킷(trim/pp/ps)별 레이블 분포
- URL 이상 여부 확인
- 폴더별 분포

실행:
  python check_distribution.py
"""

import csv
import random
import requests
from collections import Counter, defaultdict


# ── CSV 읽기 ─────────────────────────────────────────────────────────────────

records = []
with open("sample_1500.csv", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append({k: v.strip().strip('"') for k, v in row.items()})

print(f"총 샘플 수: {len(records)}")
print(f"컬럼: {list(records[0].keys()) if records else '없음'}")


# ── URL 이상 여부 확인 ───────────────────────────────────────────────────────

print("\n── URL 샘플 5개 (raw) ──")
for r in records[:5]:
    print(repr(r["image_url"]))

print("\n── URL 접근 테스트 (랜덤 5개) ──")
samples = random.sample(records, min(5, len(records)))
for r in samples:
    url = r["image_url"].strip()
    try:
        resp = requests.head(url, timeout=5)
        status = f"✅ {resp.status_code}"
    except Exception as e:
        status = f"❌ {e}"
    print(f"  {status} | {url.split('/')[-1]}")


# ── 버킷별 레이블 분포 ───────────────────────────────────────────────────────

# collected_by 기준으로 버킷별 레코드 분리
buckets = defaultdict(list)
for r in records:
    for cb in r.get("collected_by", "").split(","):
        cb = cb.strip()
        if cb:
            buckets[cb].append(r)

print("\n" + "="*50)
for bucket in ["trim", "pp", "ps"]:
    bucket_records = buckets.get(bucket, [])
    print(f"\n{'='*50}")
    print(f"  버킷: [{bucket}]  ({len(bucket_records)}장)")
    print(f"{'='*50}")

    if not bucket_records:
        print("  (없음)")
        continue

    # trim 분포
    trim_counter = Counter(r["trim_pred"].strip() for r in bucket_records)
    total = sum(trim_counter.values())
    print(f"\n  [trim_pred] total={total}")
    for cls, cnt in trim_counter.most_common():
        bar = "█" * max(1, cnt // 5)
        print(f"    {cls:15s} {cnt:5d} ({cnt/total*100:5.1f}%) {bar}")

    # pattern_size 분포
    ps_counter = Counter(r["ps_pred"].strip() for r in bucket_records)
    total = sum(ps_counter.values())
    print(f"\n  [pattern_size] total={total}")
    for cls, cnt in ps_counter.most_common():
        bar = "█" * max(1, cnt // 5)
        print(f"    {cls:15s} {cnt:5d} ({cnt/total*100:5.1f}%) {bar}")

    # pattern_position 분포
    pp_counter = Counter()
    for r in bucket_records:
        for cls in r["pp_pred"].split(","):
            pp_counter[cls.strip()] += 1
    total = sum(pp_counter.values())
    print(f"\n  [pattern_position] total={total}")
    for cls, cnt in pp_counter.most_common():
        bar = "█" * max(1, cnt // 5)
        print(f"    {cls:15s} {cnt:5d} ({cnt/total*100:5.1f}%) {bar}")

    # 폴더 분포
    folder_counter = Counter(r["folder"].strip() for r in bucket_records)
    print(f"\n  [folder] total={len(bucket_records)}")
    for folder, cnt in folder_counter.most_common():
        bar = "█" * max(1, cnt // 3)
        print(f"    {folder:20s} {cnt:4d} ({cnt/len(bucket_records)*100:5.1f}%) {bar}")


# ── 전체 합산 분포 ───────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"  전체 합산 ({len(records)}장)")
print(f"{'='*50}")

trim_counter = Counter(r["trim_pred"].strip() for r in records)
ps_counter   = Counter(r["ps_pred"].strip()   for r in records)
pp_counter   = Counter()
for r in records:
    for cls in r["pp_pred"].split(","):
        pp_counter[cls.strip()] += 1
folder_counter = Counter(r["folder"].strip() for r in records)

for name, counter in [("trim", trim_counter), ("pattern_size", ps_counter),
                       ("pattern_position", pp_counter), ("folder", folder_counter)]:
    total = sum(counter.values())
    print(f"\n  [{name}] total={total}")
    for cls, cnt in counter.most_common():
        bar = "█" * max(1, cnt // 10)
        print(f"    {cls:20s} {cnt:5d} ({cnt/total*100:5.1f}%) {bar}")