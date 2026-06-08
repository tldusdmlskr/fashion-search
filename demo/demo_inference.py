"""
분류기 시연 스크립트
--------------------
이미지 URL 입력 → 속성 추론 → 콘솔 출력 (Qdrant 저장 시뮬레이션)

실행:
  pip install open-clip-torch requests pillow
  python demo_inference.py
  python demo_inference.py --url https://...  (URL 직접 지정)
  python demo_inference.py --url https://... --bottom  (하의)
"""

import argparse
import io
import sys
import torch
import torch.nn as nn
import requests
from PIL import Image


# ── 설정 ────────────────────────────────────────────────────────────────────

V4_MODEL_PATH     = r"C:\Users\USER\Downloads\FashionCLIP\fashion_classifier_v4.pt"
BOTTOM_MODEL_PATH = r"C:\Users\USER\Downloads\FashionCLIP\fashion_classifier_bottom.pt"

EMBED_DIM    = 768
PP_THRESHOLD = 0.5

PP_CLASSES     = ["none", "allover", "front", "back", "hem", "upper", "side", "sleeve"]
PS_CLASSES     = ["none", "tiny", "medium", "large"]
TRIM_CLASSES   = ["plain", "banded", "mixed", "rolled", "unknown"]
LENGTH_CLASSES = ["발목", "미디", "초숏", "숏", "판별불가", "카프리", "맥시", "버뮤다"]
WAIST_CLASSES  = ["하이웨이스트", "normal", "판별불가"]


# ── 모델 ─────────────────────────────────────────────────────────────────────

class FashionAttributeClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters(): p.requires_grad = False
        self.classifier_pp   = nn.Sequential(nn.Linear(EMBED_DIM, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(PP_CLASSES)))
        self.classifier_ps   = nn.Sequential(nn.Linear(EMBED_DIM, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, len(PS_CLASSES)))
        self.classifier_trim = nn.Sequential(nn.Linear(EMBED_DIM, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(TRIM_CLASSES)))

    def forward(self, x):
        with torch.no_grad():
            emb = self.clip.encode_image(x).float()
        return {"pp": self.classifier_pp(emb), "ps": self.classifier_ps(emb), "trim": self.classifier_trim(emb)}


class BottomClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters(): p.requires_grad = False
        self.classifier_length = nn.Sequential(nn.Linear(EMBED_DIM, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(LENGTH_CLASSES)))
        self.classifier_waist  = nn.Sequential(nn.Linear(EMBED_DIM, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, len(WAIST_CLASSES)))

    def forward(self, x):
        with torch.no_grad():
            emb = self.clip.encode_image(x).float()
        return {"length": self.classifier_length(emb), "waist": self.classifier_waist(emb)}


# ── 출력 헬퍼 ────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
GRAY   = "\033[90m"
WHITE  = "\033[97m"
BLUE   = "\033[94m"

def bar(val, width=20):
    filled = round(val * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {val*100:.1f}%"

def print_result(result, image_id="demo_image"):
    print()
    print(f"{BOLD}{'─'*52}{RESET}")
    print(f"{BOLD}{CYAN}  ✦ Classifier Output{RESET}")
    print(f"{BOLD}{'─'*52}{RESET}")

    print(f"\n{BOLD}  Image ID{RESET}  {GRAY}{image_id}{RESET}")
    print()

    print(f"{BOLD}  ┌ Attributes ──────────────────────────────┐{RESET}")

    pp = result["pattern_position"]
    print(f"  │  {YELLOW}Pattern Position{RESET}  {WHITE}{', '.join(pp)}{RESET}")
    print(f"  │  {YELLOW}Pattern Size    {RESET}  {WHITE}{result['pattern_size']}{RESET}")
    print(f"  │  {YELLOW}Trim            {RESET}  {WHITE}{result['trim']}{RESET}")

    if "bottom_length" in result:
        print(f"  │  {YELLOW}Bottom Length   {RESET}  {WHITE}{result['bottom_length']}{RESET}")
        print(f"  │  {YELLOW}Waist Rise      {RESET}  {WHITE}{result['bottom_waist_rise']}{RESET}")

    print(f"{BOLD}  └──────────────────────────────────────────┘{RESET}")

    print(f"\n{BOLD}  ┌ Confidence ──────────────────────────────┐{RESET}")
    conf = result["confidence"]
    for k, v in conf.items():
        label = {
            "pp": "Pattern pos ", "ps": "Pattern size",
            "trim": "Trim        ", "length": "Length      ", "waist": "Waist rise  "
        }.get(k, k)
        color = GREEN if v >= 0.8 else YELLOW if v >= 0.6 else GRAY
        print(f"  │  {GRAY}{label}{RESET}  {color}{bar(v)}{RESET}")
    print(f"{BOLD}  └──────────────────────────────────────────┘{RESET}")

    print(f"\n{BOLD}  ┌ Qdrant Payload (simulated) ──────────────┐{RESET}")
    payload = {
        "pattern_position": result["pattern_position"],
        "pattern_size":     result["pattern_size"],
        "trim":             result["trim"],
    }
    if "bottom_length" in result:
        payload["bottom_length"]     = result["bottom_length"]
        payload["bottom_waist_rise"] = result["bottom_waist_rise"]

    import json
    payload_str = json.dumps(payload, ensure_ascii=False, indent=4)
    for line in payload_str.split("\n"):
        print(f"  │  {BLUE}{line}{RESET}")
    print(f"{BOLD}  └──────────────────────────────────────────┘{RESET}")
    print(f"\n  {GREEN}✔ Payload saved to Qdrant vector DB{RESET}")
    print()


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",    type=str, default=None)
    parser.add_argument("--bottom", action="store_true", help="하의 이미지")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{BOLD}[Init]{RESET} device: {device}")

    print(f"{BOLD}[Init]{RESET} marqo-fashionSigLIP 로드 중...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
    clip_model = clip_model.to(device)

    model = FashionAttributeClassifier(clip_model).to(device)
    model.load_state_dict(torch.load(V4_MODEL_PATH, map_location=device))
    model.eval()

    bottom_model = BottomClassifier(clip_model).to(device)
    bottom_model.load_state_dict(torch.load(BOTTOM_MODEL_PATH, map_location=device))
    bottom_model.eval()
    print(f"{BOLD}[Init]{RESET} 모델 로드 완료\n")

    while True:
        if args.url:
            url = args.url
            is_bottom = args.bottom
            args.url = None  # 다음 루프에서 입력 받기
        else:
            print(f"{GRAY}{'─'*52}{RESET}")
            url = input(f"  Image URL (또는 'q' 종료): ").strip()
            if url.lower() == 'q':
                print(f"\n  {GRAY}종료합니다.{RESET}\n")
                break
            if not url:
                continue
            ans = input(f"  하의 이미지? (y/n, 기본 n): ").strip().lower()
            is_bottom = ans == 'y'

        print(f"\n{GRAY}  이미지 로드 중...{RESET}")
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"  {YELLOW}이미지 로드 실패: {e}{RESET}")
            continue

        tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)

        pp_probs   = torch.sigmoid(out["pp"])[0]
        ps_probs   = torch.softmax(out["ps"],   dim=1)[0]
        trim_probs = torch.softmax(out["trim"], dim=1)[0]

        pp_labels = [PP_CLASSES[i] for i, v in enumerate(pp_probs) if v > PP_THRESHOLD] or ["none"]

        result = {
            "pattern_position": pp_labels,
            "pattern_size":     PS_CLASSES[int(ps_probs.argmax())],
            "trim":             TRIM_CLASSES[int(trim_probs.argmax())],
            "confidence": {
                "pp":   round(float(pp_probs.max()), 3),
                "ps":   round(float(ps_probs.max()),  3),
                "trim": round(float(trim_probs.max()), 3),
            }
        }

        if is_bottom:
            with torch.no_grad():
                bout = bottom_model(tensor)
            len_probs   = torch.softmax(bout["length"], dim=1)[0]
            waist_probs = torch.softmax(bout["waist"],  dim=1)[0]
            result["bottom_length"]     = LENGTH_CLASSES[int(len_probs.argmax())]
            result["bottom_waist_rise"] = WAIST_CLASSES[int(waist_probs.argmax())]
            result["confidence"]["length"] = round(float(len_probs.max()),   3)
            result["confidence"]["waist"]  = round(float(waist_probs.max()), 3)

        image_id = url.split("/")[-1].split(".")[0]
        print_result(result, image_id)

        if args.bottom is not None:
            break


if __name__ == "__main__":
    main()