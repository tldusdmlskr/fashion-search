"""
하의 속성 분류기 학습 (bottom_classifier)
------------------------------------------
  bottom_length / bottom_waist_rise 두 개 분류기
  백본: marqo-fashionSigLIP (embed_dim 768, 공유)
  waist_rise: 미드웨이스트 + 로우라이즈 → normal
  희귀 length 클래스 오버샘플링
  project-29: 발목/미디 언더샘플링 (기존 분포 맞춤)

실행:
  pip install open-clip-torch
  venv\Scripts\activate
  python train_bottom_classifier.py
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
from sklearn.metrics import classification_report, confusion_matrix


# ── 설정 ────────────────────────────────────────────────────────────────────

JSON_PATHS = [
    r"C:\Users\USER\Downloads\FashionCLIP\K-fasion 분류기 학습 파이프라인\project-26-at-2026-06-02-11-37-5ffd2a8e.json",
    
]
IMAGE_DIR  = Path(r"C:\Users\USER\Downloads\FashionCLIP\images")
MODEL_SAVE = r"C:\Users\USER\Downloads\FashionCLIP\fashion_classifier_bottom.pt"

MASKING_BASE = "https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev/masking_data"
EMBED_DIM    = 768

# project-29에서 다수 클래스 언더샘플 한도
# 기존 750장 분포: 발목 285, 미디 182 → 비율 맞춰 제한
P29_UNDERSAMPLE = {
    "발목": 40,   # 80장 → 40장
    "미디": 25,   # 51장 → 25장
}
UNDERSAMPLE_FILE = "project-29"


# ── 레이블 정의 ──────────────────────────────────────────────────────────────

LENGTH_CLASSES = ["발목", "미디", "초숏", "숏", "판별불가", "카프리", "맥시", "버뮤다"]

WAIST_MERGE = {
    "미드웨이스트 — 자연 허리~골반":     "normal",
    "로우라이즈 — 자연 허리보다 아래":   "normal",
    "하이웨이스트 — 배꼽 위·자연 허리":  "하이웨이스트",
    "판별불가 — 허리 가림·판단 불가":    "판별불가",
}
WAIST_CLASSES = ["하이웨이스트", "normal", "판별불가"]

LENGTH_MERGE = {
    "발목 — 발목뼈 근처":           "발목",
    "미디 — 정강이 중간":           "미디",
    "초숏 — 허벅지 상단":           "초숏",
    "숏 — 허벅지 중간":             "숏",
    "판별불가 — 기장 판단 불가":    "판별불가",
    "카프리 — 종아리 중간":         "카프리",
    "맥시 — 바닥·신발 덮는 길이":   "맥시",
    "버뮤다 — 무릎·무릎 바로 위":   "버뮤다",
}

LENGTH_LABEL2IDX = {cls: i for i, cls in enumerate(LENGTH_CLASSES)}
WAIST_LABEL2IDX  = {cls: i for i, cls in enumerate(WAIST_CLASSES)}

NUM_LENGTH = len(LENGTH_CLASSES)
NUM_WAIST  = len(WAIST_CLASSES)

# 오버샘플링 타겟 (희귀 클래스)
LENGTH_OVERSAMPLE_TARGET = {
    "맥시":   100,
    "버뮤다": 100,
    "카프리": 100,
    "숏":     100,
    "초숏":   100,
}
RARE_THRESHOLD = 50


# ── Augmentation ─────────────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
])

rare_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
])


# ── JSON 파싱 ────────────────────────────────────────────────────────────────

def parse_label_studio_json(json_paths: list[str]) -> list[dict]:
    all_records = []

    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        fname          = Path(path).name
        is_undersample = UNDERSAMPLE_FILE in fname
        cnts           = {cls: 0 for cls in P29_UNDERSAMPLE} if is_undersample else {}

        items = data.copy()
        if is_undersample:
            random.shuffle(items)

        valid, skipped = 0, 0
        for item in items:
            if not item["annotations"]: continue
            ann = item["annotations"][0]
            if ann["was_cancelled"]: continue

            length_raw, waist_raw = None, None
            for r in ann["result"]:
                fn = r.get("from_name")
                if fn == "bottom_length":
                    length_raw = r["value"]["choices"][0]
                elif fn == "bottom_waist_rise":
                    waist_raw = r["value"]["choices"][0]

            if None in (length_raw, waist_raw): continue

            length_cls = LENGTH_MERGE.get(length_raw)
            waist_cls  = WAIST_MERGE.get(waist_raw)

            if length_cls not in LENGTH_LABEL2IDX: continue
            if waist_cls  not in WAIST_LABEL2IDX:  continue

            # project-29 언더샘플링
            if is_undersample and length_cls in P29_UNDERSAMPLE:
                if cnts[length_cls] >= P29_UNDERSAMPLE[length_cls]:
                    skipped += 1
                    continue
                cnts[length_cls] += 1

            file_id     = str(item["data"]["file_id"])
            image_url   = item["data"]["image"]
            masking_url = f"{MASKING_BASE}/{file_id}_bottom.jpg"

            all_records.append({
                "image_url":    image_url,
                "masking_url":  masking_url,
                "file_id":      file_id,
                "length":       LENGTH_LABEL2IDX[length_cls],
                "waist":        WAIST_LABEL2IDX[waist_cls],
                "length_name":  length_cls,
                "waist_name":   waist_cls,
                "augment":      False,
            })
            valid += 1

        msg = f"→ 유효: {valid}"
        if is_undersample:
            msg += f" (언더샘플로 {skipped}개 제외 | {cnts})"
        print(f"[Parse] {fname} {msg}")

    print(f"[Parse] 전체 유효: {len(all_records)}")
    length_dist = Counter(r["length_name"] for r in all_records)
    waist_dist  = Counter(r["waist_name"]  for r in all_records)
    print(f"[Parse] length: {dict(length_dist.most_common())}")
    print(f"[Parse] waist:  {dict(waist_dist.most_common())}")
    return all_records


# ── 오버샘플링 ───────────────────────────────────────────────────────────────

def oversample(records: list[dict]) -> list[dict]:
    length_counter = Counter(r["length_name"] for r in records)
    augmented = []

    for rec in records:
        cls = rec["length_name"]
        if cls in LENGTH_OVERSAMPLE_TARGET:
            target = LENGTH_OVERSAMPLE_TARGET[cls]
            repeat = max(0, round(target / max(length_counter[cls], 1)) - 1)
            for _ in range(repeat):
                augmented.append({**rec, "augment": True})
        elif length_counter[cls] <= RARE_THRESHOLD:
            repeat = max(1, RARE_THRESHOLD // max(length_counter[cls], 1))
            for _ in range(repeat):
                augmented.append({**rec, "augment": True})

    original = records.copy()
    result   = original + augmented
    print(f"[Oversample] 원본: {len(original)} / 추가: {len(augmented)} / 최종: {len(result)}")
    after = Counter(r["length_name"] for r in result)
    print(f"[Oversample] length 후: {dict(after.most_common())}")
    return result


# ── 클래스 가중치 ────────────────────────────────────────────────────────────

def compute_class_weights(records: list[dict]) -> dict:
    length_counter = Counter(r["length_name"] for r in records)
    length_total   = sum(length_counter.values())
    length_weights = torch.tensor(
        [math.log1p(length_total / (NUM_LENGTH * length_counter.get(cls, 1)))
         for cls in LENGTH_CLASSES], dtype=torch.float32)

    waist_counter = Counter(r["waist_name"] for r in records)
    waist_total   = sum(waist_counter.values())
    waist_weights = torch.tensor(
        [math.log1p(waist_total / (NUM_WAIST * waist_counter.get(cls, 1)))
         for cls in WAIST_CLASSES], dtype=torch.float32)

    print(f"[Weights] length: {[round(v,2) for v in length_weights.tolist()]}")
    print(f"[Weights] waist:  {[round(v,2) for v in waist_weights.tolist()]}")
    return {"length": length_weights, "waist": waist_weights}


# ── 이미지 로드 ──────────────────────────────────────────────────────────────

def load_image(file_id: str, image_url: str, masking_url: str) -> Image.Image:
    local = IMAGE_DIR / masking_url.split("/")[-1]
    if local.exists():
        return Image.open(local).convert("RGB")
    local = IMAGE_DIR / f"{file_id}.jpg"
    if local.exists():
        return Image.open(local).convert("RGB")
    try:
        resp = requests.get(masking_url, timeout=10)
        if resp.status_code == 200:
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


# ── Dataset ──────────────────────────────────────────────────────────────────

class BottomDataset(Dataset):
    def __init__(self, records: list[dict], preprocess, is_train: bool = True):
        self.records    = records
        self.preprocess = preprocess
        self.is_train   = is_train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            image = load_image(rec["file_id"], rec["image_url"], rec["masking_url"])
        except Exception as e:
            print(f"[Dataset] 로드 실패: {rec['file_id']} → {e}")
            image = Image.new("RGB", (224, 224))

        if rec.get("augment"):
            image = rare_augment(image)
        elif self.is_train:
            image = train_transform(image)

        return {
            "image":  self.preprocess(image),
            "length": torch.tensor(rec["length"], dtype=torch.long),
            "waist":  torch.tensor(rec["waist"],  dtype=torch.long),
        }


# ── 모델 ─────────────────────────────────────────────────────────────────────

class BottomClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False

        self.classifier_length = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, NUM_LENGTH),
        )
        self.classifier_waist = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, NUM_WAIST),
        )

    def forward(self, images):
        with torch.no_grad():
            emb = self.clip.encode_image(images).float()
        return {
            "length": self.classifier_length(emb),
            "waist":  self.classifier_waist(emb),
        }


# ── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        print(f"  [EarlyStopping] 개선 없음 {self.counter}/{self.patience}")
        return self.counter >= self.patience


# ── Confusion Matrix ─────────────────────────────────────────────────────────

def print_cm(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    print(f"\n── {title} Confusion Matrix ──")
    print(f"  {'':12s}" + "".join(f"{l:>8s}" for l in labels))
    for i, label in enumerate(labels):
        print(f"  {label:12s}" + "".join(f"{cm[i][j]:>8d}" for j in range(len(labels))))


# ── 학습 ─────────────────────────────────────────────────────────────────────

def train(json_paths: list[str], epochs: int = 50, batch_size: int = 16, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device: {device}")

    print("[Train] marqo-fashionSigLIP 로드 중...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    clip_model = clip_model.to(device)
    print("[Train] 로드 완료")

    records = parse_label_studio_json(json_paths)
    records = oversample(records)
    weights = compute_class_weights(records)

    train_rec, val_rec = train_test_split(records, test_size=0.2, random_state=42)
    print(f"[Train] train: {len(train_rec)} / val: {len(val_rec)}")

    train_loader = DataLoader(BottomDataset(train_rec, preprocess, is_train=True),
                              batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(BottomDataset(val_rec,   preprocess, is_train=False),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    model      = BottomClassifier(clip_model).to(device)
    optimizer  = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    early_stop = EarlyStopping(patience=5, min_delta=0.001)

    ce_length = nn.CrossEntropyLoss(weight=weights["length"].to(device))
    ce_waist  = nn.CrossEntropyLoss(weight=weights["waist"].to(device))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = (
                ce_length(outputs["length"], batch["length"].to(device)) +
                ce_waist(outputs["waist"],   batch["waist"].to(device))
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        model.eval()
        len_correct = waist_correct = total = 0
        val_loss_sum = 0
        len_preds, len_labels_all = [], []
        waist_preds, waist_labels_all = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                outputs = model(images)
                total += images.size(0)
                v_loss = (
                    ce_length(outputs["length"], batch["length"].to(device)) +
                    ce_waist(outputs["waist"],   batch["waist"].to(device))
                )
                val_loss_sum += v_loss.item()
                len_correct   += (outputs["length"].argmax(1) == batch["length"].to(device)).sum().item()
                waist_correct += (outputs["waist"].argmax(1)  == batch["waist"].to(device)).sum().item()
                len_preds.extend(outputs["length"].argmax(1).cpu().tolist())
                len_labels_all.extend(batch["length"].tolist())
                waist_preds.extend(outputs["waist"].argmax(1).cpu().tolist())
                waist_labels_all.extend(batch["waist"].tolist())

        avg_val = val_loss_sum / len(val_loader)
        print(f"Epoch {epoch:02d} | train: {avg_train:.4f} | val: {avg_val:.4f} | "
              f"length: {len_correct/total*100:.1f}% | waist: {waist_correct/total*100:.1f}%")

        if early_stop.step(avg_val, model):
            print(f"\n[EarlyStopping] 학습 중단 | best: {early_stop.best_loss:.4f}")
            model.load_state_dict(early_stop.best_state)
            break

    if early_stop.best_state:
        model.load_state_dict(early_stop.best_state)
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"\n✅ 모델 저장: {MODEL_SAVE}")

    print("\n── Classification Report ──")
    print("\n[bottom_length]")
    print(classification_report(len_labels_all, len_preds, target_names=LENGTH_CLASSES, zero_division=0))
    print_cm(len_labels_all, len_preds, LENGTH_CLASSES, "bottom_length")

    print("\n[bottom_waist_rise]")
    print(classification_report(waist_labels_all, waist_preds, target_names=WAIST_CLASSES, zero_division=0))
    print_cm(waist_labels_all, waist_preds, WAIST_CLASSES, "bottom_waist_rise")

    return model


# ── 실행 진입점 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = train(
        json_paths=JSON_PATHS,
        epochs=50,
        batch_size=16,
        lr=1e-4,
    )