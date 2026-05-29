import json

with open("project-19-at-2026-05-24-18-40-2c0325ed.json", encoding="utf-8") as f:
    data = json.load(f)

has_pp, has_ps, has_trim = 0, 0, 0
has_all, has_pp_only = 0, 0

for item in data:
    if not item["annotations"]:
        continue
    ann = item["annotations"][0]
    if ann["was_cancelled"]:
        continue
    fns = {r.get("from_name") for r in ann["result"]}
    if "pattern_position" in fns: has_pp += 1
    if "pattern_size" in fns: has_ps += 1
    if "trim" in fns: has_trim += 1
    if all(x in fns for x in ("pattern_position", "pattern_size", "trim")): has_all += 1
    if "pattern_position" in fns and "pattern_size" not in fns and "trim" not in fns: has_pp_only += 1

print(f"pattern_position만 있는 샘플: {has_pp_only}")
print(f"pattern_position 있는 샘플:  {has_pp}")
print(f"pattern_size 있는 샘플:      {has_ps}")
print(f"trim 있는 샘플:              {has_trim}")
print(f"3개 모두 있는 샘플:           {has_all}")