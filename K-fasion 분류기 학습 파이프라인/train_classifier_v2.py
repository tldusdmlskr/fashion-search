"""
K-Fashion 속성 분류기 학습 (v2)
---------------------------------
  두 JSON 파일 합산 + 속성 통합:
    trim    : raw+frayed → raw / standard+hemmed → standard
    pp      : leg 제거
  불균형 처리:
    - 다수 클래스(none/standard) 언더샘플링
    - trim 희귀 클래스(rolled/unknown/mixed) 타겟 오버샘플링
    - loss: log 스케일 클래스 가중치

실행:
  venv\Scripts\activate        ← Windows
  source venv/bin/activate     ← Mac/Linux
  python train_classifier.py
"""

import json
import io
import math
import random
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


# ── 레이블 정의 ──────────────────────────────────────────────────────────────

PP_CLASSES = ["none", "allover", "front", "back", "hem",
              "neckline", "shoulder", "side", "sleeve"]
PP2IDX = {cls: i for i, cls in enumerate(PP_CLASSES)}

TRIM_MERGE = {
    "frayed": "raw",
    "hemmed": "standard",
}

SINGLE_CLASSES = {
    "pattern_size": ["none", "tiny", "small", "medium", "large"],
    "trim":         ["standard", "banded", "raw", "mixed", "rolled", "unknown"],
}
SINGLE_LABEL2IDX = {
    task: {cls: i for i, cls in enumerate(classes)}
    for task, classes in SINGLE_CLASSES.items()
}
NUM_SINGLE_CLASSES = {task: len(c) for task, c in SINGLE_CLASSES.items()}


# ── 샘플링 설정 ──────────────────────────────────────────────────────────────

# 언더샘플링: 다수 클래스 최대 수 제한
UNDERSAMPLE_LIMIT = {
    "pp_none":        700,
    "ps_none":        700,
    "trim_standard":  700,
}

# 타겟 오버샘플링: trim 희귀 클래스별 목표 장수
TRIM_OVERSAMPLE_TARGET = {
    "rolled":  300,   # 95장 → ~285장
    "unknown": 150,   # 52장 → ~156장
    "mixed":   150,   # 52장 → ~156장
}

# pp/ps 오버샘플링 기준
PP_PS_RARE_THRESHOLD = 50


# ── Augmentation ─────────────────────────────────────────────────────────────

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

rare_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Label Studio JSON 파싱 ───────────────────────────────────────────────────

def parse_label_studio_json(json_paths: list[str]) -> list[dict]:
    all_data = []
    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        all_data.extend(data)
        print(f"[Parse] {path} → {len(data)}개 항목")

    records = []
    for item in all_data:
        if not item["annotations"]:
            continue
        ann = item["annotations"][0]
        if ann["was_cancelled"]:
            continue

        pp_choices, ps_choice, trim_choice = None, None, None
        for r in ann["result"]:
            fn = r.get("from_name")
            if fn == "pattern_position":
                pp_choices = r["value"]["choices"]
            elif fn == "pattern_size":
                ps_choice = r["value"]["choices"][0]
            elif fn == "trim":
                trim_choice = r["value"]["choices"][0]

        if None in (pp_choices, ps_choice, trim_choice):
            continue

        trim_choice = TRIM_MERGE.get(trim_choice, trim_choice)
        if trim_choice not in SINGLE_LABEL2IDX["trim"]:
            continue

        pp_choices = [c for c in pp_choices if c != "leg"] or ["none"]

        pp_multihot = [0.0] * len(PP_CLASSES)
        for cls in pp_choices:
            if cls in PP2IDX:
                pp_multihot[PP2IDX[cls]] = 1.0

        records.append({
            "image_url":        item["data"]["image"],
            "file_id":          item["data"]["file_id"],
            "pattern_position": pp_multihot,
            "pattern_size":     SINGLE_LABEL2IDX["pattern_size"][ps_choice],
            "trim":             SINGLE_LABEL2IDX["trim"][trim_choice],
            "pp_choices":       pp_choices,
            "ps_name":          ps_choice,
            "trim_name":        trim_choice,
        })

    print(f"[Parse] 전체: {len(all_data)} / 유효: {len(records)}")
    return records


# ── 언더샘플링 ───────────────────────────────────────────────────────────────

def undersample(records: list[dict]) -> list[dict]:
    result = []
    pp_none_cnt = 0
    ps_none_cnt = 0
    trim_std_cnt = 0

    shuffled = records.copy()
    random.shuffle(shuffled)

    for rec in shuffled:
        pp_is_none_only  = rec["pp_choices"] == ["none"]
        ps_is_none       = rec["ps_name"] == "none"
        trim_is_standard = rec["trim_name"] == "standard"

        if pp_is_none_only:
            if pp_none_cnt >= UNDERSAMPLE_LIMIT["pp_none"]:
                continue
            pp_none_cnt += 1

        if ps_is_none:
            if ps_none_cnt >= UNDERSAMPLE_LIMIT["ps_none"]:
                continue
            ps_none_cnt += 1

        if trim_is_standard:
            if trim_std_cnt >= UNDERSAMPLE_LIMIT["trim_standard"]:
                continue
            trim_std_cnt += 1

        result.append(rec)

    print(f"[Undersample] {len(records)} → {len(result)}")
    print(f"             pp_none: {pp_none_cnt} | ps_none: {ps_none_cnt} | trim_standard: {trim_std_cnt}")
    return result


# ── 오버샘플링 ───────────────────────────────────────────────────────────────

def oversample_all(records: list[dict]) -> list[dict]:
    pp_counter   = Counter()
    ps_counter   = Counter()
    trim_counter = Counter()

    for rec in records:
        for cls in rec["pp_choices"]:
            if cls in PP2IDX:
                pp_counter[cls] += 1
        ps_counter[rec["ps_name"]] += 1
        trim_counter[rec["trim_name"]] += 1

    # pp/ps 희귀 클래스
    pp_rare = {cls for cls, cnt in pp_counter.items() if cnt <= PP_PS_RARE_THRESHOLD}
    ps_rare = {cls for cls, cnt in ps_counter.items() if cnt <= PP_PS_RARE_THRESHOLD}

    print(f"[Oversample] pp   희귀 (≤{PP_PS_RARE_THRESHOLD}): {pp_rare}")
    print(f"[Oversample] ps   희귀 (≤{PP_PS_RARE_THRESHOLD}): {ps_rare}")
    print(f"[Oversample] trim 타겟: {TRIM_OVERSAMPLE_TARGET}")

    augmented = []
    for rec in records:
        repeats = []

        # pp 희귀
        if any(cls in pp_rare for cls in rec["pp_choices"]):
            min_cnt = min(pp_counter[cls] for cls in rec["pp_choices"] if cls in pp_rare)
            repeats.append(max(1, PP_PS_RARE_THRESHOLD // max(min_cnt, 1)))

        # ps 희귀
        if rec["ps_name"] in ps_rare:
            repeats.append(max(1, PP_PS_RARE_THRESHOLD // max(ps_counter[rec["ps_name"]], 1)))

        # trim 타겟 오버샘플링
        trim_name = rec["trim_name"]
        if trim_name in TRIM_OVERSAMPLE_TARGET:
            target = TRIM_OVERSAMPLE_TARGET[trim_name]
            current = trim_counter[trim_name]
            repeat = max(0, round(target / max(current, 1)) - 1)
            if repeat > 0:
                repeats.append(repeat)

        if not repeats:
            continue

        for _ in range(max(repeats)):
            augmented.append({**rec, "augment": True})

    original = [{**rec, "augment": False} for rec in records]
    result = original + augmented

    # 오버샘플링 후 분포 확인
    after_trim = Counter(r["trim_name"] for r in result)
    print(f"[Oversample] 원본: {len(original)} / 추가: {len(augmented)} / 최종: {len(result)}")
    print(f"[Oversample] trim 후: {dict(after_trim.most_common())}")
    return result


# ── 클래스 가중치 ────────────────────────────────────────────────────────────

def compute_class_weights(records: list[dict]) -> dict:
    ps_counter = Counter(rec["ps_name"] for rec in records)
    ps_total   = sum(ps_counter.values())
    ps_weights = torch.tensor(
        [math.log1p(ps_total / (len(SINGLE_CLASSES["pattern_size"]) * ps_counter.get(cls, 1)))
         for cls in SINGLE_CLASSES["pattern_size"]],
        dtype=torch.float32,
    )

    tr_counter = Counter(rec["trim_name"] for rec in records)
    tr_total   = sum(tr_counter.values())
    tr_weights = torch.tensor(
        [math.log1p(tr_total / (len(SINGLE_CLASSES["trim"]) * tr_counter.get(cls, 1)))
         for cls in SINGLE_CLASSES["trim"]],
        dtype=torch.float32,
    )

    pp_counter = Counter()
    for rec in records:
        for i, v in enumerate(rec["pattern_position"]):
            if v == 1.0:
                pp_counter[i] += 1
    n = len(records)
    pp_pos_weight = torch.tensor(
        [math.log1p((n - pp_counter.get(i, 1)) / max(pp_counter.get(i, 1), 1))
         for i in range(len(PP_CLASSES))],
        dtype=torch.float32,
    )

    print(f"[Weights] pattern_size: {[round(v,2) for v in ps_weights.tolist()]}")
    print(f"[Weights] trim:         {[round(v,2) for v in tr_weights.tolist()]}")
    print(f"[Weights] pp pos_weight: {[round(v,2) for v in pp_pos_weight.tolist()]}")
    return {"pattern_size": ps_weights, "trim": tr_weights, "pp_pos_weight": pp_pos_weight}


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_image(url: str) -> Image.Image:
    file_id = url.rstrip("/").split("/")[-1].split(".")[0]
    local_path = Path("images") / f"{file_id}.jpg"
    if local_path.exists():
        return Image.open(local_path).convert("RGB")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


class FashionAttributeDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            image = load_image(rec["image_url"])
        except Exception as e:
            print(f"[Dataset] 로드 실패: {rec['image_url']} → {e}")
            image = Image.new("RGB", (224, 224))
        transform = rare_transform if rec.get("augment") else base_transform
        return {
            "image":            transform(image),
            "pattern_position": torch.tensor(rec["pattern_position"], dtype=torch.float32),
            "pattern_size":     torch.tensor(rec["pattern_size"],     dtype=torch.long),
            "trim":             torch.tensor(rec["trim"],             dtype=torch.long),
        }


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
            nn.Linear(64, NUM_SINGLE_CLASSES["pattern_size"]),
        )
        self.classifier_trim = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, NUM_SINGLE_CLASSES["trim"]),
        )

    def forward(self, images):
        with torch.no_grad():
            embeddings = self.clip.encode_image(images).float()
        return {
            "pattern_position": self.classifier_pattern_position(embeddings),
            "pattern_size":     self.classifier_pattern_size(embeddings),
            "trim":             self.classifier_trim(embeddings),
        }


# ── 학습 ─────────────────────────────────────────────────────────────────────

def train(json_paths: list[str], epochs: int = 20, batch_size: int = 32, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device: {device}")

    print("[Train] CLIP 로드 중...")
    import clip
    clip_model, _ = clip.load("ViT-B/32", device=device)
    print("[Train] CLIP 로드 완료")

    records = parse_label_studio_json(json_paths)
    records = undersample(records)
    records = oversample_all(records)
    weights = compute_class_weights(records)

    train_rec, val_rec = train_test_split(records, test_size=0.2, random_state=42)
    print(f"[Train] train: {len(train_rec)} / val: {len(val_rec)}")

    train_loader = DataLoader(FashionAttributeDataset(train_rec),
                              batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(FashionAttributeDataset(val_rec),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    model     = FashionAttributeClassifier(clip_model).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=weights["pp_pos_weight"].to(device))
    ce_ps    = nn.CrossEntropyLoss(weight=weights["pattern_size"].to(device))
    ce_trim  = nn.CrossEntropyLoss(weight=weights["trim"].to(device))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = (
                bce_loss(outputs["pattern_position"], batch["pattern_position"].to(device)) +
                ce_ps(outputs["pattern_size"],        batch["pattern_size"].to(device)) +
                ce_trim(outputs["trim"],              batch["trim"].to(device))
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        pp_preds, pp_labels = [], []
        ps_correct, tr_correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                outputs = model(images)
                total += images.size(0)
                pp_pred = (torch.sigmoid(outputs["pattern_position"]) > 0.5).cpu().int().tolist()
                pp_true = batch["pattern_position"].int().tolist()
                pp_preds.extend(pp_pred)
                pp_labels.extend(pp_true)
                ps_correct += (outputs["pattern_size"].argmax(1) == batch["pattern_size"].to(device)).sum().item()
                tr_correct += (outputs["trim"].argmax(1)         == batch["trim"].to(device)).sum().item()

        pp_f1 = f1_score(pp_labels, pp_preds, average="micro", zero_division=0)
        print(f"Epoch {epoch:02d} | loss: {avg_loss:.4f} | "
              f"pp F1(micro): {pp_f1:.3f} | "
              f"pattern_size: {ps_correct/total*100:.1f}% | "
              f"trim: {tr_correct/total*100:.1f}%")

    torch.save(model.state_dict(), "fashion_classifier_v2.pt")
    print("\n✅ 모델 저장 완료: fashion_classifier_v2.pt")

    print("\n── Classification Report ──")
    model.eval()
    pp_preds, pp_labels = [], []
    ps_preds, ps_labels = [], []
    tr_preds, tr_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            outputs = model(images)
            pp_preds.extend((torch.sigmoid(outputs["pattern_position"]) > 0.5).cpu().int().tolist())
            pp_labels.extend(batch["pattern_position"].int().tolist())
            ps_preds.extend(outputs["pattern_size"].argmax(1).cpu().tolist())
            ps_labels.extend(batch["pattern_size"].tolist())
            tr_preds.extend(outputs["trim"].argmax(1).cpu().tolist())
            tr_labels.extend(batch["trim"].tolist())

    print("\n[pattern_position] (멀티레이블, per-class F1)")
    print(classification_report(pp_labels, pp_preds, target_names=PP_CLASSES, zero_division=0))
    print("\n[pattern_size]")
    print(classification_report(ps_labels, ps_preds,
                                target_names=SINGLE_CLASSES["pattern_size"], zero_division=0))
    print("\n[trim]")
    print(classification_report(tr_labels, tr_preds,
                                target_names=SINGLE_CLASSES["trim"], zero_division=0))

    return model


# ── 실행 진입점 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    JSON_PATHS = [
        "project-19-at-2026-05-29-00-55-58e3457a.json",
        "project-21-at-2026-05-29-00-56-95af36e6.json",
    ]

    model = train(
        json_paths=JSON_PATHS,
        epochs=20,
        batch_size=32,
        lr=1e-4,
    )