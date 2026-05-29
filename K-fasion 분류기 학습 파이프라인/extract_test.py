"""
10만장 균형 샘플링
-----------------------------------
세 분류기 각각 희귀 클래스 500장씩 수집:
  - trim    : standard 제외 500장 (banded 최대 100장)
  - pp      : none 제외 500장
  - ps      : none 제외 500장
  중복 이미지 실시간 제거 → 최종 최대 1,500장

실행 전 환경변수 설정 (PowerShell):
  $env:R2_ENDPOINT   = "https://<계정ID>.r2.cloudflarestorage.com"
  $env:R2_ACCESS_KEY = "..."
  $env:R2_SECRET_KEY = "..."

실행:
  python sample_for_labeling.py
"""

import io
import os
import csv
import random
import requests
import boto3
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── 설정 ────────────────────────────────────────────────────────────────────

R2_ENDPOINT   = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET     = "project2"

R2_PUBLIC_BASE = "https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev/image"
MODEL_PATH     = r"C:\Users\USER\Downloads\FashionCLIP\fashion_classifier_v1.pt"  # ★ 실제 경로

OUTPUT_SAMPLE    = "sample_1500.csv"
TARGET_PER_TASK  = 500
BANDED_LIMIT     = 100    # trim 버킷에서 banded 최대 수집 장수
SAVE_EVERY       = 300
BATCH_SIZE       = 32
DOWNLOAD_WORKERS = 8

TRIM_SKIP = {"standard"}
PP_SKIP   = {"none"}
PS_SKIP   = {"none"}

FOLDERS = [
    "avant_garde", "classic", "country", "etc", "feminine", "genderless",
    "hiphop", "hippie", "kitsch", "mannish", "military", "modern",
    "oriental", "preppy", "punk", "resort", "retro", "romantic",
    "sexy", "sophisticated", "sporty", "street", "tomboy", "western",
]

PP_CLASSES = ["none", "allover", "front", "back", "hem", "leg",
              "neckline", "shoulder", "side", "sleeve"]
SINGLE_CLASSES = {
    "pattern_size": ["none", "tiny", "small", "medium", "large"],
    "trim":         ["standard", "banded", "frayed", "hemmed",
                     "mixed", "raw", "rolled", "unknown"],
}

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

CSV_FIELDS = ["image_id", "folder", "image_url",
              "pp_pred", "ps_pred", "trim_pred", "collected_by"]


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
            embeddings = self.clip.encode_image(images).float()
        return {
            "pattern_position": self.classifier_pattern_position(embeddings),
            "pattern_size":     self.classifier_pattern_size(embeddings),
            "trim":             self.classifier_trim(embeddings),
        }


def load_model(model_path: str, device):
    print("[Model] CLIP 로드 중...")
    import clip
    clip_model, _ = clip.load("ViT-B/32", device=device)
    model = FashionAttributeClassifier(clip_model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[Model] 로드 완료")
    return model


# ── R2 이미지 목록 수집 ──────────────────────────────────────────────────────

def list_images_from_r2() -> list[dict]:
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
        folder_records = []
        for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith(".jpg"):
                    continue
                image_id = Path(key).stem
                folder_records.append({
                    "image_id":  image_id,
                    "folder":    folder,
                    "image_url": f"{R2_PUBLIC_BASE}/{folder}/{image_id}.jpg",
                })
        print(f"  [{folder}] {len(folder_records):,}장")
        records.extend(folder_records)

    random.shuffle(records)
    print(f"[List] 총 {len(records):,}장 (랜덤 셔플됨)")
    return records


# ── 이미지 다운로드 ──────────────────────────────────────────────────────────

def fetch_image(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url.strip(), timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


# ── 균형 샘플링 ──────────────────────────────────────────────────────────────

def collect_balanced(model, device, records: list[dict], target: int) -> list[dict]:
    trim_bucket  = {}   # image_id → True (카운팅용)
    pp_bucket    = {}
    ps_bucket    = {}
    banded_count = 0    # trim 버킷 내 banded 카운터

    all_ids     = {}    # image_id → row (중복 없는 전체 저장)
    save_buffer = []
    processed   = 0

    # CSV 초기화
    with open(OUTPUT_SAMPLE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()

    def all_done():
        return (len(trim_bucket) >= target and
                len(pp_bucket)   >= target and
                len(ps_bucket)   >= target)

    def status_str():
        return (f"trim: {len(trim_bucket):,}/{target}(banded:{banded_count}) | "
                f"pp: {len(pp_bucket):,}/{target} | "
                f"ps: {len(ps_bucket):,}/{target} | "
                f"총: {len(all_ids):,}")

    for start in range(0, len(records), BATCH_SIZE):
        if all_done():
            print(f"\n[Done] 세 버킷 모두 {target}장 달성 → 중단")
            break

        batch = records[start:start + BATCH_SIZE]

        images, valid = [], []
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            future_map = {executor.submit(fetch_image, r["image_url"]): r for r in batch}
            for future in as_completed(future_map):
                rec = future_map[future]
                img = future.result()
                if img is not None:
                    images.append(base_transform(img))
                    valid.append(rec)

        if not images:
            continue

        tensor = torch.stack(images).to(device)
        with torch.no_grad():
            outputs = model(tensor)

        pp_probs   = torch.sigmoid(outputs["pattern_position"])
        pp_preds   = (pp_probs > 0.5).cpu().int().tolist()
        ps_preds   = outputs["pattern_size"].argmax(1).cpu().tolist()
        trim_preds = outputs["trim"].argmax(1).cpu().tolist()

        for i, rec in enumerate(valid):
            iid = rec["image_id"]

            pp_labels = [PP_CLASSES[j] for j, v in enumerate(pp_preds[i]) if v == 1]
            ps_label  = SINGLE_CLASSES["pattern_size"][ps_preds[i]]
            tr_label  = SINGLE_CLASSES["trim"][trim_preds[i]]
            pp_str    = ",".join(pp_labels) if pp_labels else "none"

            collected_by = []

            # trim 버킷: standard 제외 + banded 100장 제한
            if len(trim_bucket) < target and tr_label not in TRIM_SKIP:
                if tr_label == "banded" and banded_count >= BANDED_LIMIT:
                    pass  # banded 한도 초과 → 스킵
                elif iid not in trim_bucket:
                    trim_bucket[iid] = True
                    if tr_label == "banded":
                        banded_count += 1
                    collected_by.append("trim")

            # pp 버킷: none 제외
            if len(pp_bucket) < target:
                pp_non_none = [c for c in pp_labels if c not in PP_SKIP]
                if pp_non_none and iid not in pp_bucket:
                    pp_bucket[iid] = True
                    collected_by.append("pp")

            # ps 버킷: none 제외
            if len(ps_bucket) < target and ps_label not in PS_SKIP:
                if iid not in ps_bucket:
                    ps_bucket[iid] = True
                    collected_by.append("ps")

            # 하나라도 수집됐고 전체 저장 안 된 이미지면 저장
            if collected_by and iid not in all_ids:
                row = {
                    "image_id":    iid,
                    "folder":      rec["folder"],
                    "image_url":   rec["image_url"].strip(),
                    "pp_pred":     pp_str,
                    "ps_pred":     ps_label,
                    "trim_pred":   tr_label,
                    "collected_by": ",".join(collected_by),
                }
                all_ids[iid] = row
                save_buffer.append(row)

        processed += len(valid)

        if len(save_buffer) >= SAVE_EVERY:
            with open(OUTPUT_SAMPLE, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
                writer.writerows(save_buffer)
            save_buffer = []
            print(f"[Infer] 처리: {processed:,}장 | {status_str()} | 저장 완료")
        else:
            print(f"[Infer] 처리: {processed:,}장 | {status_str()}", end="\r")

    # 남은 버퍼 저장
    if save_buffer:
        with open(OUTPUT_SAMPLE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
            writer.writerows(save_buffer)

    return list(all_ids.values())


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] device: {device}")

    model = load_model(MODEL_PATH, device)

    print("\n[List] R2 버킷 이미지 목록 수집 중...")
    all_records = list_images_from_r2()

    print(f"\n[Infer] 균형 샘플링 시작 (버킷별 {TARGET_PER_TASK}장, banded 최대 {BANDED_LIMIT}장)")
    collected = collect_balanced(model, device, all_records, TARGET_PER_TASK)

    # 최종 CSV 덮어쓰기
    with open(OUTPUT_SAMPLE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(collected)

    print(f"\n✅ 완료: {len(collected)}장 → {OUTPUT_SAMPLE}")
    print("   Label Studio 담당자한테 이 파일 넘기면 돼요!")


if __name__ == "__main__":
    main()