"""
Confidence 낮은 샘플 추출 (v4 기준, 태스크별 100장 + 폴더 비율)
-----------------------------------------------------------------
  pp / ps / trim 각각 confidence 낮은 100장씩 추출
  각 태스크 안에서 폴더(스타일) 비율 맞춰 샘플링
  중복 제거 후 최종 300장

실행:
  $env:R2_ENDPOINT   = "https://..."
  $env:R2_ACCESS_KEY = "..."
  $env:R2_SECRET_KEY = "..."
  python extract_low_confidence.py
"""

import os
import io
import csv
import json
import random
import torch
import torch.nn as nn
import requests
import boto3
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── 설정 ────────────────────────────────────────────────────────────────────

R2_ENDPOINT    = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY  = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY  = os.environ["R2_SECRET_KEY"]
R2_BUCKET      = "project2"
R2_PUBLIC_BASE = "https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev/image"

MODEL_PATH  = r"C:\Users\USER\Downloads\FashionCLIP\fashion_classifier_v4.pt"
OUTPUT_CSV  = r"C:\Users\USER\Downloads\FashionCLIP\low_confidence_300.csv"

PER_TASK    = 100    # 태스크별 추출 장수
FOLDER_LIMIT = 15   # 폴더당 최대 장수 (100장 / 24폴더 ≈ 4장, 여유있게 15)
BATCH_SIZE  = 32
WORKERS     = 8
INFER_LIMIT = 10000  # 최대 추론 장수 (충분히 많이 추론 후 정렬)

EXISTING_JSON_PATHS = [
    r"C:\Users\USER\Downloads\FashionCLIP\K-fasion 분류기 학습 파이프라인\project-19-at-2026-05-29-00-55-58e3457a.json",
    r"C:\Users\USER\Downloads\FashionCLIP\K-fasion 분류기 학습 파이프라인\project-21-at-2026-05-29-00-56-95af36e6.json",
    r"C:\Users\USER\Downloads\FashionCLIP\K-fasion 분류기 학습 파이프라인\project-24-at-2026-05-31-00-47-c0c7e362.json",
]

EMBED_DIM = 768

FOLDERS = [
    "avant_garde", "classic", "country", "etc", "feminine", "genderless",
    "hiphop", "hippie", "kitsch", "mannish", "military", "modern",
    "oriental", "preppy", "punk", "resort", "retro", "romantic",
    "sexy", "sophisticated", "sporty", "street", "tomboy", "western",
]

PP_CLASSES   = ["none", "allover", "front", "back", "hem", "upper", "side", "sleeve"]
PS_CLASSES   = ["none", "tiny", "medium", "large"]
TRIM_CLASSES = ["plain", "banded", "mixed", "rolled", "unknown"]

CSV_FIELDS = [
    "image_id", "folder", "image_url",
    "pp_pred", "ps_pred", "trim_pred",
    "pp_conf", "ps_conf", "trim_conf",
    "sampled_by",   # 어떤 태스크로 뽑혔는지
]


# ── 모델 ─────────────────────────────────────────────────────────────────────

class FashionAttributeClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False
        self.classifier_pp = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, len(PP_CLASSES)),
        )
        self.classifier_ps = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, len(PS_CLASSES)),
        )
        self.classifier_trim = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, len(TRIM_CLASSES)),
        )

    def forward(self, images):
        with torch.no_grad():
            emb = self.clip.encode_image(images).float()
        return {
            "pp":   self.classifier_pp(emb),
            "ps":   self.classifier_ps(emb),
            "trim": self.classifier_trim(emb),
        }


# ── 기존 라벨링 ID ───────────────────────────────────────────────────────────

def load_existing_ids(json_paths):
    ids = set()
    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            ids.add(str(item["data"]["file_id"]))
    print(f"[Existing] 기존 라벨링 제외: {len(ids)}장")
    return ids


# ── R2 목록 수집 ─────────────────────────────────────────────────────────────

def list_images(existing_ids):
    client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )
    records = []
    for folder in FOLDERS:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=f"image/{folder}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith(".jpg"): continue
                image_id = Path(key).stem
                if image_id in existing_ids: continue
                records.append({
                    "image_id":  image_id,
                    "folder":    folder,
                    "image_url": f"{R2_PUBLIC_BASE}/{folder}/{image_id}.jpg",
                })
    random.shuffle(records)
    print(f"[List] 총 {len(records):,}장")
    return records


# ── 이미지 다운로드 ──────────────────────────────────────────────────────────

def fetch_image(url):
    try:
        resp = requests.get(url.strip(), timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


# ── 추론 ─────────────────────────────────────────────────────────────────────

def run_inference(model, preprocess, device, records):
    all_results = []
    processed   = 0

    for start in range(0, min(len(records), INFER_LIMIT), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        images, valid = [], []

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            future_map = {executor.submit(fetch_image, r["image_url"]): r for r in batch}
            for future in as_completed(future_map):
                rec = future_map[future]
                img = future.result()
                if img is not None:
                    images.append(preprocess(img))
                    valid.append(rec)

        if not images: continue

        tensor = torch.stack(images).to(device)
        with torch.no_grad():
            outputs = model(tensor)

        pp_sigmoid   = torch.sigmoid(outputs["pp"]).cpu()
        ps_softmax   = torch.softmax(outputs["ps"],   dim=1).cpu()
        trim_softmax = torch.softmax(outputs["trim"], dim=1).cpu()

        for i, rec in enumerate(valid):
            pp_probs  = pp_sigmoid[i]
            pp_pred   = [PP_CLASSES[j] for j, v in enumerate(pp_probs) if v > 0.5]
            pp_conf   = float(pp_probs.max())

            ps_prob   = ps_softmax[i]
            ps_idx    = int(ps_prob.argmax())
            ps_conf   = float(ps_prob.max())

            trim_prob = trim_softmax[i]
            trim_idx  = int(trim_prob.argmax())
            trim_conf = float(trim_prob.max())

            all_results.append({
                "image_id":  rec["image_id"],
                "folder":    rec["folder"],
                "image_url": rec["image_url"],
                "pp_pred":   ",".join(pp_pred) if pp_pred else "none",
                "ps_pred":   PS_CLASSES[ps_idx],
                "trim_pred": TRIM_CLASSES[trim_idx],
                "pp_conf":   round(pp_conf,   4),
                "ps_conf":   round(ps_conf,   4),
                "trim_conf": round(trim_conf, 4),
            })

        processed += len(valid)
        print(f"[Infer] {processed:,}/{min(len(records), INFER_LIMIT):,}장 처리", end="\r")

    print(f"\n[Infer] 추론 완료: {len(all_results):,}장")
    return all_results


# ── 태스크별 폴더 비율 맞춰 샘플링 ──────────────────────────────────────────

def sample_by_task(all_results, task: str, n: int) -> list[dict]:
    """
    task 기준 confidence 낮은 순 정렬 후
    폴더 비율 맞춰 n장 샘플링
    """
    conf_key = f"{task}_conf"
    sorted_results = sorted(all_results, key=lambda x: x[conf_key])

    # 전체 폴더 비율 계산
    folder_total = Counter(r["folder"] for r in all_results)
    total        = sum(folder_total.values())

    # 폴더별 할당
    base_quota = n // len(FOLDERS)
    remainder  = n % len(FOLDERS)
    folder_quota = {}
    for i, folder in enumerate(FOLDERS):
        ratio = folder_total.get(folder, 0) / max(total, 1)
        quota = min(round(ratio * n), FOLDER_LIMIT)
        folder_quota[folder] = max(1, quota) if folder_total.get(folder, 0) > 0 else 0

    # 폴더별 그룹화 (confidence 낮은 순 이미 정렬됨)
    folder_groups = defaultdict(list)
    for r in sorted_results:
        folder_groups[r["folder"]].append(r)

    # 폴더별 quota만큼 앞에서 뽑기
    sampled = []
    for folder, quota in folder_quota.items():
        pool = folder_groups.get(folder, [])
        sampled.extend(pool[:quota])

    # 부족하면 나머지에서 confidence 낮은 순으로 추가
    if len(sampled) < n:
        sampled_ids = {r["image_id"] for r in sampled}
        rest = [r for r in sorted_results if r["image_id"] not in sampled_ids]
        sampled.extend(rest[:n - len(sampled)])

    return sampled[:n]


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] device: {device}")

    print("[Main] marqo-fashionSigLIP 로드 중...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    clip_model = clip_model.to(device)

    model = FashionAttributeClassifier(clip_model).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[Main] 모델 로드: {MODEL_PATH}")

    existing_ids = load_existing_ids(EXISTING_JSON_PATHS)
    records      = list_images(existing_ids)
    all_results  = run_inference(model, preprocess, device, records)

    # 태스크별 100장씩 샘플링
    print("\n[Sample] 태스크별 샘플링 중...")
    pp_samples   = sample_by_task(all_results, "pp",   PER_TASK)
    ps_samples   = sample_by_task(all_results, "ps",   PER_TASK)
    trim_samples = sample_by_task(all_results, "trim", PER_TASK)

    print(f"  pp   : {len(pp_samples)}장")
    print(f"  ps   : {len(ps_samples)}장")
    print(f"  trim : {len(trim_samples)}장")

    # 중복 제거하면서 합치기
    final = []
    seen  = set()

    for task, samples in [("pp", pp_samples), ("ps", ps_samples), ("trim", trim_samples)]:
        for r in samples:
            if r["image_id"] not in seen:
                seen.add(r["image_id"])
                final.append({**r, "sampled_by": task})

    # 중복 제거로 300장 미만이면 나머지에서 추가
    if len(final) < PER_TASK * 3:
        need = PER_TASK * 3 - len(final)
        rest = sorted(
            [r for r in all_results if r["image_id"] not in seen],
            key=lambda x: x["pp_conf"] + x["ps_conf"] + x["trim_conf"]
        )
        for r in rest[:need]:
            final.append({**r, "sampled_by": "low_total"})
        print(f"[Sample] 중복 제거 후 부족분 {need}장 추가")

    random.shuffle(final)

    # 분포 출력
    folder_dist = Counter(r["folder"] for r in final)
    task_dist   = Counter(r["sampled_by"] for r in final)
    print(f"\n── 최종 {len(final)}장 ──")
    print(f"태스크별: {dict(task_dist)}")
    print(f"\n폴더별 분포:")
    for folder, cnt in folder_dist.most_common():
        bar = "█" * cnt
        print(f"  {folder:22s} {cnt:3d} {bar}")

    # confidence 분포
    for task in ["pp", "ps", "trim"]:
        confs = [r[f"{task}_conf"] for r in final]
        print(f"\n{task} conf — min: {min(confs):.3f} / mean: {sum(confs)/len(confs):.3f} / max: {max(confs):.3f}")

    # CSV 저장
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(final)

    print(f"\n✅ 완료: {OUTPUT_CSV} ({len(final)}장)")


if __name__ == "__main__":
    main()