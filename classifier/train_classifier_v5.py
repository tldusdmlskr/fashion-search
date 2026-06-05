"""
K-Fashion 속성 분류기 학습 (v4 - marqo-fashionSigLIP)
------------------------------------------------------
  속성 통합:
    trim: standard + raw + frayed + hemmed → plain
    pp:   neckline + shoulder → upper / leg 제거
    ps:   small → medium
  project-24: 기존 라벨러(ID 1,4,5,6)만 사용
  project-25: 전체 사용 (ID 1, 4만 작업)
  공유 백본 + 분류기 헤드 3개 동시 학습

실행:
  pip install open-clip-torch
  venv\Scripts\activate
  python train_classifier_v4.py
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
from sklearn.metrics import (classification_report, f1_score,
                              confusion_matrix, multilabel_confusion_matrix)


# ── 설정 ────────────────────────────────────────────────────────────────────

JSON_PATHS = [
    r"C:\Users\USER\Downloads\FashionCLIP\K-fasion 분류기 학습 파이프라인\project-19-at-2026-05-29-00-55-58e3457a.json",
    r"C:\Users\USER\Downloads\FashionCLIP\K-fasion 분류기 학습 파이프라인\project-21-at-2026-05-29-00-56-95af36e6.json",
    r"C:\Users\USER\Downloads\FashionCLIP\K-fasion 분류기 학습 파이프라인\project-24-at-2026-05-31-00-47-c0c7e362.json",
    r"C:\Users\USER\Downloads\FashionCLIP\K-fasion 분류기 학습 파이프라인\project-25-at-2026-06-02-01-04-c611442a.json",
]
IMAGE_DIR  = Path(r"C:\Users\USER\Downloads\FashionCLIP\images")
MODEL_SAVE = r"C:\Users\USER\Downloads\FashionCLIP\fashion_classifier_v4.pt"

MASKING_BASE = "https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev/masking_data"
TYPE_MAP = {
    "상의":   "top",
    "하의":   "bottom",
    "아우터": "outerwear",
    "원피스": "dress",
}

EMBED_DIM = 768

# project-24만 annotator 필터링
VALID_ANNOTATORS  = {1, 4, 5, 6}
FILTERED_FILES    = {"project-24"}   # 이 파일들만 필터 적용


# ── 레이블 정의 ──────────────────────────────────────────────────────────────

PP_MERGE = {"neckline": "upper", "shoulder": "upper"}
PP_CLASSES = ["none", "allover", "front", "back", "hem",
              "upper", "side", "sleeve"]
PP2IDX = {cls: i for i, cls in enumerate(PP_CLASSES)}

TRIM_MERGE = {
    "standard": "plain",
    "raw":      "plain",
    "frayed":   "plain",
    "hemmed":   "plain",
}
PS_MERGE = {"small": "medium"}

SINGLE_CLASSES = {
    "pattern_size": ["none", "tiny", "medium", "large"],
    "trim":         ["plain", "banded", "mixed", "rolled", "unknown"],
}
SINGLE_LABEL2IDX = {
    task: {cls: i for i, cls in enumerate(classes)}
    for task, classes in SINGLE_CLASSES.items()
}
NUM_SINGLE_CLASSES = {task: len(c) for task, c in SINGLE_CLASSES.items()}


# ── 샘플링 설정 ──────────────────────────────────────────────────────────────

UNDERSAMPLE_LIMIT = {
    "pp_none":     700,
    "ps_none":     700,
    "ps_medium":   600,
    "trim_plain":  700,
}

TRIM_OVERSAMPLE_TARGET = {"rolled": 300, "unknown": 150, "mixed": 150}
PS_OVERSAMPLE_TARGET   = {"tiny": 300, "large": 300}
PP_PS_RARE_THRESHOLD   = 50


# ── Augmentation ─────────────────────────────────────────────────────────────

rare_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
])


# ── clothing_type 추출 ───────────────────────────────────────────────────────

def extract_clothing_type(item_data: dict) -> str | None:
    if "clothing_type" in item_data:
        return item_data["clothing_type"].strip()
    info = item_data.get("item_info", "")
    for line in info.split("\n"):
        if line.startswith("타입:"):
            return line.replace("타입:", "").strip().split("/")[0].strip()
    return None


# ── JSON 파싱 ────────────────────────────────────────────────────────────────

def parse_label_studio_json(json_paths: list[str]) -> list[dict]:
    all_records = []

    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        fname       = Path(path).name
        is_filtered = any(f in fname for f in FILTERED_FILES)
        valid, skipped = 0, 0

        for item in data:
            if not item["annotations"]: continue
            ann = item["annotations"][0]
            if ann["was_cancelled"]: continue

            if is_filtered:
                cb = ann.get("completed_by")
                if isinstance(cb, dict): cb = cb.get("id")
                if cb not in VALID_ANNOTATORS:
                    skipped += 1
                    continue

            pp_choices, ps_choice, trim_choice = None, None, None
            for r in ann["result"]:
                fn = r.get("from_name")
                if fn == "pattern_position": pp_choices = r["value"]["choices"]
                elif fn == "pattern_size":   ps_choice  = r["value"]["choices"][0]
                elif fn == "trim":           trim_choice = r["value"]["choices"][0]

            if None in (pp_choices, ps_choice, trim_choice): continue

            trim_choice = TRIM_MERGE.get(trim_choice, trim_choice)
            if trim_choice not in SINGLE_LABEL2IDX["trim"]: continue

            ps_choice = PS_MERGE.get(ps_choice, ps_choice)
            if ps_choice not in SINGLE_LABEL2IDX["pattern_size"]: continue

            pp_choices = [c for c in pp_choices if c != "leg"]
            pp_choices = list(set(PP_MERGE.get(c, c) for c in pp_choices)) or ["none"]

            pp_multihot = [0.0] * len(PP_CLASSES)
            for cls in pp_choices:
                if cls in PP2IDX: pp_multihot[PP2IDX[cls]] = 1.0

            file_id       = str(item["data"]["file_id"])
            image_url     = item["data"]["image"]
            clothing_type = extract_clothing_type(item["data"])

            masking_url = None
            if clothing_type and clothing_type in TYPE_MAP:
                masking_url = f"{MASKING_BASE}/{file_id}_{TYPE_MAP[clothing_type]}.jpg"

            all_records.append({
                "image_url":        image_url,
                "masking_url":      masking_url,
                "file_id":          file_id,
                "pattern_position": pp_multihot,
                "pattern_size":     SINGLE_LABEL2IDX["pattern_size"][ps_choice],
                "trim":             SINGLE_LABEL2IDX["trim"][trim_choice],
                "pp_choices":       pp_choices,
                "ps_name":          ps_choice,
                "trim_name":        trim_choice,
            })
            valid += 1

        msg = f"→ 유효: {valid}"
        if is_filtered: msg += f" (annotator 필터로 {skipped}개 제외)"
        print(f"[Parse] {fname} {msg}")

    print(f"[Parse] 전체 유효: {len(all_records)}")
    trim_dist = Counter(r["trim_name"] for r in all_records)
    ps_dist   = Counter(r["ps_name"]   for r in all_records)
    print(f"[Parse] trim: {dict(trim_dist.most_common())}")
    print(f"[Parse] ps:   {dict(ps_dist.most_common())}")
    return all_records


# ── 언더샘플링 ───────────────────────────────────────────────────────────────

def undersample(records: list[dict]) -> list[dict]:
    result = []
    cnts   = {k: 0 for k in UNDERSAMPLE_LIMIT}
    shuffled = records.copy()
    random.shuffle(shuffled)

    for rec in shuffled:
        if rec["pp_choices"] == ["none"]:
            if cnts["pp_none"] >= UNDERSAMPLE_LIMIT["pp_none"]: continue
            cnts["pp_none"] += 1
        if rec["ps_name"] == "none":
            if cnts["ps_none"] >= UNDERSAMPLE_LIMIT["ps_none"]: continue
            cnts["ps_none"] += 1
        if rec["ps_name"] == "medium":
            if cnts["ps_medium"] >= UNDERSAMPLE_LIMIT["ps_medium"]: continue
            cnts["ps_medium"] += 1
        if rec["trim_name"] == "plain":
            if cnts["trim_plain"] >= UNDERSAMPLE_LIMIT["trim_plain"]: continue
            cnts["trim_plain"] += 1
        result.append(rec)

    print(f"[Undersample] {len(records)} → {len(result)} | {cnts}")
    return result


# ── 오버샘플링 ───────────────────────────────────────────────────────────────

def oversample_all(records: list[dict]) -> list[dict]:
    ps_counter   = Counter(r["ps_name"]   for r in records)
    trim_counter = Counter(r["trim_name"] for r in records)
    ps_rare      = {cls for cls, cnt in ps_counter.items() if cnt <= PP_PS_RARE_THRESHOLD}

    print(f"[Oversample] ps   희귀 (≤{PP_PS_RARE_THRESHOLD}): {ps_rare}")
    print(f"[Oversample] ps   타겟: {PS_OVERSAMPLE_TARGET}")
    print(f"[Oversample] trim 타겟: {TRIM_OVERSAMPLE_TARGET}")

    augmented = []
    for rec in records:
        repeats = []
        ps_name = rec["ps_name"]
        if ps_name in PS_OVERSAMPLE_TARGET:
            repeat = max(0, round(PS_OVERSAMPLE_TARGET[ps_name] / max(ps_counter[ps_name], 1)) - 1)
            if repeat > 0: repeats.append(repeat)
        elif ps_name in ps_rare:
            repeats.append(max(1, PP_PS_RARE_THRESHOLD // max(ps_counter[ps_name], 1)))

        trim_name = rec["trim_name"]
        if trim_name in TRIM_OVERSAMPLE_TARGET:
            repeat = max(0, round(TRIM_OVERSAMPLE_TARGET[trim_name] / max(trim_counter[trim_name], 1)) - 1)
            if repeat > 0: repeats.append(repeat)

        if not repeats: continue
        for _ in range(max(repeats)):
            augmented.append({**rec, "augment": True})

    original = [{**rec, "augment": False} for rec in records]
    result   = original + augmented
    print(f"[Oversample] 원본: {len(original)} / 추가: {len(augmented)} / 최종: {len(result)}")
    print(f"[Oversample] trim 후: {dict(Counter(r['trim_name'] for r in result).most_common())}")
    print(f"[Oversample] ps 후:   {dict(Counter(r['ps_name']   for r in result).most_common())}")
    return result


# ── 클래스 가중치 ────────────────────────────────────────────────────────────

def compute_class_weights(records: list[dict]) -> dict:
    pp_counter = Counter()
    for rec in records:
        for i, v in enumerate(rec["pattern_position"]):
            if v == 1.0: pp_counter[i] += 1
    n = len(records)
    pp_pos_weight = torch.tensor(
        [math.log1p((n - pp_counter.get(i, 1)) / max(pp_counter.get(i, 1), 1))
         for i in range(len(PP_CLASSES))], dtype=torch.float32)

    ps_counter = Counter(rec["ps_name"] for rec in records)
    ps_total   = sum(ps_counter.values())
    ps_weights = torch.tensor(
        [math.log1p(ps_total / (len(SINGLE_CLASSES["pattern_size"]) * ps_counter.get(cls, 1)))
         for cls in SINGLE_CLASSES["pattern_size"]], dtype=torch.float32)

    tr_counter = Counter(rec["trim_name"] for rec in records)
    tr_total   = sum(tr_counter.values())
    tr_weights = torch.tensor(
        [math.log1p(tr_total / (len(SINGLE_CLASSES["trim"]) * tr_counter.get(cls, 1)))
         for cls in SINGLE_CLASSES["trim"]], dtype=torch.float32)

    print(f"[Weights] pp pos_weight: {[round(v,2) for v in pp_pos_weight.tolist()]}")
    print(f"[Weights] ps:            {[round(v,2) for v in ps_weights.tolist()]}")
    print(f"[Weights] trim:          {[round(v,2) for v in tr_weights.tolist()]}")
    return {"pp_pos_weight": pp_pos_weight, "ps": ps_weights, "trim": tr_weights}


# ── 이미지 로드 ──────────────────────────────────────────────────────────────

def load_image(file_id: str, image_url: str, masking_url: str | None) -> Image.Image:
    if masking_url:
        local = IMAGE_DIR / masking_url.split("/")[-1]
        if local.exists():
            return Image.open(local).convert("RGB")
    local = IMAGE_DIR / f"{file_id}.jpg"
    if local.exists():
        return Image.open(local).convert("RGB")
    if masking_url:
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

class FashionAttributeDataset(Dataset):
    def __init__(self, records: list[dict], preprocess):
        self.records    = records
        self.preprocess = preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            image = load_image(rec["file_id"], rec["image_url"], rec.get("masking_url"))
        except Exception as e:
            print(f"[Dataset] 로드 실패: {rec['file_id']} → {e}")
            image = Image.new("RGB", (224, 224))

        if rec.get("augment"):
            image = rare_augment(image)

        return {
            "image":            self.preprocess(image),
            "pattern_position": torch.tensor(rec["pattern_position"], dtype=torch.float32),
            "pattern_size":     torch.tensor(rec["pattern_size"],     dtype=torch.long),
            "trim":             torch.tensor(rec["trim"],             dtype=torch.long),
        }


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
            nn.Linear(128, NUM_SINGLE_CLASSES["pattern_size"]),
        )
        self.classifier_trim = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, NUM_SINGLE_CLASSES["trim"]),
        )

    def forward(self, images):
        with torch.no_grad():
            emb = self.clip.encode_image(images).float()
        return {
            "pattern_position": self.classifier_pp(emb),
            "pattern_size":     self.classifier_ps(emb),
            "trim":             self.classifier_trim(emb),
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
    print(f"  {'':12s}" + "".join(f"{l:>10s}" for l in labels))
    for i, label in enumerate(labels):
        print(f"  {label:12s}" + "".join(f"{cm[i][j]:>10d}" for j in range(len(labels))))


def print_pp_cm(y_true, y_pred, labels):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    print(f"\n── pattern_position Confusion Matrix ──")
    print(f"  {'클래스':12s}  {'TN':>6}  {'FP':>6}  {'FN':>6}  {'TP':>6}")
    for i, label in enumerate(labels):
        tn, fp, fn, tp = mcm[i].ravel()
        print(f"  {label:12s}  {tn:6d}  {fp:6d}  {fn:6d}  {tp:6d}")


# ── 학습 ─────────────────────────────────────────────────────────────────────

def train(json_paths: list[str], epochs: int = 50, batch_size: int = 32, lr: float = 1e-4):
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
    records = undersample(records)
    records = oversample_all(records)
    weights = compute_class_weights(records)

    train_rec, val_rec = train_test_split(records, test_size=0.2, random_state=42)
    print(f"[Train] train: {len(train_rec)} / val: {len(val_rec)}")

    train_loader = DataLoader(FashionAttributeDataset(train_rec, preprocess),
                              batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(FashionAttributeDataset(val_rec,   preprocess),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    model      = FashionAttributeClassifier(clip_model).to(device)
    optimizer  = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    early_stop = EarlyStopping(patience=5, min_delta=0.001)

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=weights["pp_pos_weight"].to(device))
    ce_ps    = nn.CrossEntropyLoss(weight=weights["ps"].to(device))
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

        avg_train = total_loss / len(train_loader)
        model.eval()
        pp_preds, pp_labels = [], []
        ps_correct, tr_correct, total = 0, 0, 0
        val_loss_sum = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                outputs = model(images)
                total += images.size(0)
                v_loss = (
                    bce_loss(outputs["pattern_position"], batch["pattern_position"].to(device)) +
                    ce_ps(outputs["pattern_size"],        batch["pattern_size"].to(device)) +
                    ce_trim(outputs["trim"],              batch["trim"].to(device))
                )
                val_loss_sum += v_loss.item()
                pp_pred = (torch.sigmoid(outputs["pattern_position"]) > 0.5).cpu().int().tolist()
                pp_true = batch["pattern_position"].int().tolist()
                pp_preds.extend(pp_pred)
                pp_labels.extend(pp_true)
                ps_correct += (outputs["pattern_size"].argmax(1) == batch["pattern_size"].to(device)).sum().item()
                tr_correct += (outputs["trim"].argmax(1)         == batch["trim"].to(device)).sum().item()

        avg_val = val_loss_sum / len(val_loader)
        pp_f1   = f1_score(pp_labels, pp_preds, average="micro", zero_division=0)

        print(f"Epoch {epoch:02d} | train: {avg_train:.4f} | val: {avg_val:.4f} | "
              f"pp F1: {pp_f1:.3f} | ps: {ps_correct/total*100:.1f}% | trim: {tr_correct/total*100:.1f}%")

        if early_stop.step(avg_val, model):
            print(f"\n[EarlyStopping] 학습 중단 | best: {early_stop.best_loss:.4f}")
            model.load_state_dict(early_stop.best_state)
            break

    if early_stop.best_state:
        model.load_state_dict(early_stop.best_state)
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"\n✅ 모델 저장: {MODEL_SAVE}")

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

    print("\n[pattern_position]")
    print(classification_report(pp_labels, pp_preds, target_names=PP_CLASSES, zero_division=0))
    print_pp_cm(pp_labels, pp_preds, PP_CLASSES)

    print("\n[pattern_size]")
    print(classification_report(ps_labels, ps_preds,
                                target_names=SINGLE_CLASSES["pattern_size"], zero_division=0))
    print_cm(ps_labels, ps_preds, SINGLE_CLASSES["pattern_size"], "pattern_size")

    print("\n[trim]")
    print(classification_report(tr_labels, tr_preds,
                                target_names=SINGLE_CLASSES["trim"], zero_division=0))
    print_cm(tr_labels, tr_preds, SINGLE_CLASSES["trim"], "trim")

    return model


# ── 실행 진입점 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = train(
        json_paths=JSON_PATHS,
        epochs=50,
        batch_size=32,
        lr=1e-4,
    )