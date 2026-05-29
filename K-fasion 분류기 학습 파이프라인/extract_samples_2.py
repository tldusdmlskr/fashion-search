"""
전체 추론 + 희귀 클래스 균형 샘플링
--------------------------------------
1. R2 전체 이미지 목록 수집 (기존 라벨링 제외)
2. 전체 추론 → inference_all.csv 저장 (체크포인트 지원)
3. 희귀 클래스 해당 이미지만 필터링
4. 스타일(폴더)별 비율 맞춰 2,000장 샘플링 → rare_samples_2000.csv

실행 전 환경변수 설정 (PowerShell):
  $env:R2_ENDPOINT   = "https://<계정ID>.r2.cloudflarestorage.com"
  $env:R2_ACCESS_KEY = "..."
  $env:R2_SECRET_KEY = "..."

실행:
  python extract_rare_samples.py
"""

import io
import os
import csv
import json
import random
import requests
import boto3
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── 설정 ────────────────────────────────────────────────────────────────────

R2_ENDPOINT   = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET     = "project2"

R2_PUBLIC_BASE = "https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev/image"
MODEL_PATH     = r"C:\Users\USER\Downloads\FashionCLIP\fashion_classifier_v2.pt"

EXISTING_JSON_PATHS = [
    r"C:\Users\USER\Downloads\FashionCLIP\project-19-at-2026-05-29-00-55-58e3457a.json",
    r"C:\Users\USER\Downloads\FashionCLIP\project-21-at-2026-05-29-00-56-95af36e6.json",
]

OUTPUT_ALL    = "inference_all.csv"       # 전체 추론 결과
OUTPUT_SAMPLE = "rare_samples_2000.csv"   # 최종 샘플링 결과

TARGET_SIZE      = 2000
FOLDER_LIMIT     = 100    # 폴더(스타일)당 최대 수집 장수
BATCH_SIZE       = 32
DOWNLOAD_WORKERS = 8
SAVE_EVERY       = 1000   # 전체 추론 중간 저장 주기

# 희귀 클래스 기준
RARE_PP   = {"back", "neckline", "shoulder", "side"}
RARE_TRIM = {"rolled", "unknown", "mixed"}
RARE_PS   = {"tiny", "large"}

FOLDERS = [
    "avant_garde", "classic", "country", "etc", "feminine", "genderless",
    "hiphop", "hippie", "kitsch", "mannish", "military", "modern",
    "oriental", "preppy", "punk", "resort", "retro", "romantic",
    "sexy", "sophisticated", "sporty", "street", "tomboy", "western",
]


# ── 레이블 정의 ──────────────────────────────────────────────────────────────

PP_CLASSES = ["none", "allover", "front", "back", "hem",
              "neckline", "shoulder", "side", "sleeve"]
SINGLE_CLASSES = {
    "pattern_size": ["none", "tiny", "small", "medium", "large"],
    "trim":         ["standard", "banded", "raw", "mixed", "rolled", "unknown"],
}

ALL_FIELDS = ["image_id", "folder", "image_url",
              "pp_pred", "ps_pred", "trim_pred", "rare_reason"]


# ── 모델 ─────────────────────────────────────────────────────────────────────

class FashionAttributeClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim: int = 512):
        super().__init__()
        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False
        self.classifier_pattern_position = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, len(PP_CLASSES)),
        )
        self.classifier_pattern_size = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, len(SINGLE_CLASSES["pattern_size"])),
        )
        self.classifier_trim = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, len(SINGLE_CLASSES["trim"])),
        )

    def forward(self, images):
        with torch.no_grad():
            embeddings = self.clip.get_image_features(pixel_values=images).float()
        return {
            "pattern_position": self.classifier_pattern_position(embeddings),
            "pattern_size":     self.classifier_pattern_size(embeddings),
            "trim":             self.classifier_trim(embeddings),
        }


def load_model(model_path: str, device):
    print("[Model] FashionCLIP 로드 중...")
    from fashion_clip.fashion_clip import FashionCLIP
    fc         = FashionCLIP("fashion-clip")
    clip_model = fc.model.to(device)
    preprocess = fc.preprocess
    model = FashionAttributeClassifier(clip_model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Model] 로드 완료: {model_path}")
    return model, preprocess


# ── 기존 라벨링 file_id 수집 ─────────────────────────────────────────────────

def load_existing_ids(json_paths: list[str]) -> set[str]:
    existing_ids = set()
    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            existing_ids.add(str(item["data"]["file_id"]))
    print(f"[Existing] 기존 라벨링 file_id: {len(existing_ids)}개")
    return existing_ids


# ── R2 이미지 목록 수집 ──────────────────────────────────────────────────────

def list_images_from_r2(existing_ids: set[str]) -> list[dict]:
    client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )
    records = []
    for folder in FOLDERS:
        prefix = f"image/{folder}/"
        paginator = client.get_paginator("list_objects_v2")
        folder_cnt = 0
        for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith(".jpg"):
                    continue
                image_id = Path(key).stem
                if image_id in existing_ids:
                    continue
                records.append({
                    "image_id":  image_id,
                    "folder":    folder,
                    "image_url": f"{R2_PUBLIC_BASE}/{folder}/{image_id}.jpg",
                })
                folder_cnt += 1
        print(f"  [{folder}] {folder_cnt:,}장")

    random.shuffle(records)
    print(f"[List] 총 {len(records):,}장 (기존 {len(existing_ids)}장 제외)")
    return records


# ── 체크포인트 로드 ──────────────────────────────────────────────────────────

def load_checkpoint() -> tuple[list[dict], set[str]]:
    if not Path(OUTPUT_ALL).exists():
        return [], set()
    done = []
    done_ids = set()
    with open(OUTPUT_ALL, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.append({k: v.strip().strip('"') for k, v in row.items()})
            done_ids.add(row["image_id"].strip().strip('"'))
    print(f"[Checkpoint] 이전 결과 {len(done):,}장 로드 → 이어서 처리")
    return done, done_ids


# ── 이미지 다운로드 ──────────────────────────────────────────────────────────

def fetch_image(url: str):
    try:
        resp = requests.get(url.strip(), timeout=10)
        resp.raise_for_status()
        from PIL import Image
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


# ── 전체 추론 ────────────────────────────────────────────────────────────────

def run_inference(model, preprocess, device, records: list[dict]) -> list[dict]:
    """전체 이미지 추론 후 결과 반환 + 체크포인트 저장"""
    results    = []
    save_buffer = []
    processed  = 0

    # 체크포인트 없으면 헤더 작성
    if not Path(OUTPUT_ALL).exists():
        with open(OUTPUT_ALL, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=ALL_FIELDS, quoting=csv.QUOTE_ALL).writeheader()

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        images, valid = [], []
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            future_map = {executor.submit(fetch_image, r["image_url"]): r for r in batch}
            for future in as_completed(future_map):
                rec = future_map[future]
                img = future.result()
                if img is not None:
                    tensor = preprocess(images=[img], return_tensors="pt")["pixel_values"][0]
                    images.append(tensor)
                    valid.append(rec)

        if not images:
            continue

        import torch
        tensor = torch.stack(images).to(device)
        with torch.no_grad():
            outputs = model(tensor)

        import torch as _torch
        pp_probs   = _torch.sigmoid(outputs["pattern_position"])
        pp_preds   = (pp_probs > 0.5).cpu().int().tolist()
        ps_preds   = outputs["pattern_size"].argmax(1).cpu().tolist()
        trim_preds = outputs["trim"].argmax(1).cpu().tolist()

        for i, rec in enumerate(valid):
            pp_labels = [PP_CLASSES[j] for j, v in enumerate(pp_preds[i]) if v == 1]
            ps_label  = SINGLE_CLASSES["pattern_size"][ps_preds[i]]
            tr_label  = SINGLE_CLASSES["trim"][trim_preds[i]]

            rare_reasons = []
            for cls in pp_labels:
                if cls in RARE_PP:
                    rare_reasons.append(f"pp:{cls}")
            if ps_label in RARE_PS:
                rare_reasons.append(f"ps:{ps_label}")
            if tr_label in RARE_TRIM:
                rare_reasons.append(f"trim:{tr_label}")

            row = {
                "image_id":    rec["image_id"],
                "folder":      rec["folder"],
                "image_url":   rec["image_url"].strip(),
                "pp_pred":     ",".join(pp_labels) if pp_labels else "none",
                "ps_pred":     ps_label,
                "trim_pred":   tr_label,
                "rare_reason": ",".join(rare_reasons),
            }
            results.append(row)
            save_buffer.append(row)

        processed += len(valid)

        if len(save_buffer) >= SAVE_EVERY:
            with open(OUTPUT_ALL, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=ALL_FIELDS, quoting=csv.QUOTE_ALL)
                writer.writerows(save_buffer)
            save_buffer = []
            print(f"[Infer] {processed:,}/{len(records):,} 완료 | 체크포인트 저장")
        else:
            print(f"[Infer] {processed:,}/{len(records):,} 완료", end="\r")

    if save_buffer:
        with open(OUTPUT_ALL, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_FIELDS, quoting=csv.QUOTE_ALL)
            writer.writerows(save_buffer)

    print(f"\n[Infer] 전체 추론 완료: {len(results):,}장 → {OUTPUT_ALL}")
    return results


# ── 폴더 비율 맞춘 샘플링 ───────────────────────────────────────────────────

def sample_with_folder_balance(all_results: list[dict], target: int) -> list[dict]:
    """
    희귀 클래스 해당 이미지 중 폴더별 비율 맞춰 target장 샘플링
    폴더당 최대 FOLDER_LIMIT장
    """
    # 희귀 클래스 해당 이미지만 필터링
    rare_records = [r for r in all_results if r["rare_reason"]]
    print(f"\n[Sample] 희귀 클래스 해당: {len(rare_records):,}장 / 전체: {len(all_results):,}장")

    # 폴더별로 그룹화
    folder_groups = defaultdict(list)
    for r in rare_records:
        folder_groups[r["folder"]].append(r)

    # 폴더별 원본 비율 계산 (R2 전체 기준)
    folder_total = Counter(r["folder"] for r in all_results)
    total = sum(folder_total.values())

    # 폴더별 할당 장수 계산 (비율 × target, 최대 FOLDER_LIMIT)
    folder_quota = {}
    for folder in FOLDERS:
        ratio = folder_total.get(folder, 0) / max(total, 1)
        quota = min(round(ratio * target), FOLDER_LIMIT)
        folder_quota[folder] = quota

    print(f"[Sample] 폴더별 할당 (상위 5개):")
    for folder, quota in sorted(folder_quota.items(), key=lambda x: -x[1])[:5]:
        avail = len(folder_groups.get(folder, []))
        print(f"  {folder:22s} 할당: {quota:3d} / 희귀 가용: {avail:4d}")

    # 폴더별 샘플링
    sampled = []
    for folder, quota in folder_quota.items():
        available = folder_groups.get(folder, [])
        n = min(quota, len(available))
        sampled.extend(random.sample(available, n))

    # 부족하면 나머지 폴더에서 추가 (FOLDER_LIMIT 무시하고 채우기)
    if len(sampled) < target:
        sampled_ids = {r["image_id"] for r in sampled}
        remaining   = [r for r in rare_records if r["image_id"] not in sampled_ids]
        random.shuffle(remaining)
        need = target - len(sampled)
        sampled.extend(remaining[:need])
        print(f"[Sample] 부족분 {need}장 추가 채움")

    random.shuffle(sampled)
    result = sampled[:target]

    # 최종 분포 출력
    folder_dist   = Counter(r["folder"]      for r in result)
    rare_dist     = Counter()
    for r in result:
        for reason in r["rare_reason"].split(","):
            rare_dist[reason.strip()] += 1

    print(f"\n[Sample] 최종 {len(result)}장 샘플링 완료")
    print(f"\n── 폴더별 분포 ──")
    for folder, cnt in folder_dist.most_common():
        bar = "█" * (cnt // 3)
        print(f"  {folder:22s} {cnt:4d} {bar}")
    print(f"\n── 희귀 클래스 분포 ──")
    for reason, cnt in rare_dist.most_common():
        bar = "█" * (cnt // 5)
        print(f"  {reason:20s} {cnt:4d} {bar}")

    return result


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] device: {device}")

    model, preprocess = load_model(MODEL_PATH, device)
    existing_ids      = load_existing_ids(EXISTING_JSON_PATHS)

    print("\n[List] R2 버킷 이미지 목록 수집 중...")
    all_records = list_images_from_r2(existing_ids)

    # 체크포인트 로드 (재실행 시 이어서)
    done_results, done_ids = load_checkpoint()
    todo = [r for r in all_records if r["image_id"] not in done_ids]
    print(f"[Main] 처리 대상: {len(todo):,}장 (완료: {len(done_ids):,}장 스킵)")

    # 전체 추론
    if todo:
        new_results  = run_inference(model, preprocess, device, todo)
        all_results  = done_results + new_results
    else:
        print("[Main] 모두 완료됨. 체크포인트에서 결과 사용.")
        all_results  = done_results

    # 폴더 비율 맞춘 2,000장 샘플링
    sampled = sample_with_folder_balance(all_results, TARGET_SIZE)

    # 최종 CSV 저장
    with open(OUTPUT_SAMPLE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sampled)

    print(f"\n✅ 완료!")
    print(f"   전체 추론 결과: {OUTPUT_ALL} ({len(all_results):,}장)")
    print(f"   샘플링 결과:    {OUTPUT_SAMPLE} ({len(sampled):,}장)")
    print(f"\nLabel Studio 담당자한테 {OUTPUT_SAMPLE} 넘기면 돼요!")


if __name__ == "__main__":
    main()