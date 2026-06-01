"""
K-Fashion 속성 분류기 추론 (v4)
--------------------------------
  새 이미지 URL 또는 PIL Image → pp/ps/trim 속성 반환
  단일 이미지 및 배치 추론 모두 지원

사용 예시:
  from inference import FashionClassifier

  clf = FashionClassifier(r"C:\...\fashion_classifier_v4.pt")

  # 단일 URL
  result = clf.predict_url("https://...")
  print(result)
  # {"pattern_position": ["allover"], "pattern_size": "medium", "trim": "banded"}

  # 배치
  results = clf.predict_batch(["url1", "url2", ...])
"""

import io
import torch
import torch.nn as nn
import requests
from pathlib import Path
from PIL import Image
from typing import Union


# ── 레이블 정의 (v4) ─────────────────────────────────────────────────────────

PP_CLASSES   = ["none", "allover", "front", "back", "hem",
                "upper", "side", "sleeve"]
PS_CLASSES   = ["none", "tiny", "medium", "large"]
TRIM_CLASSES = ["plain", "banded", "mixed", "rolled", "unknown"]

EMBED_DIM    = 768
PP_THRESHOLD = 0.5   # sigmoid threshold


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


# ── 추론 클래스 ──────────────────────────────────────────────────────────────

class FashionClassifier:
    def __init__(self, model_path: str, device: str = None):
        """
        model_path: fashion_classifier_v4.pt 경로
        device:     'cuda' / 'cpu' / None(자동)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[FashionClassifier] device: {self.device}")
        print("[FashionClassifier] marqo-fashionSigLIP 로드 중...")

        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )
        self.clip_model = clip_model.to(self.device)
        self.preprocess = preprocess

        self.model = FashionAttributeClassifier(self.clip_model).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        print(f"[FashionClassifier] 로드 완료: {model_path}")

    def _load_image(self, source: Union[str, Image.Image]) -> Image.Image:
        """URL 또는 PIL Image → PIL Image"""
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        resp = requests.get(source.strip(), timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def _postprocess(self, outputs: dict) -> dict:
        """모델 출력 → 레이블 딕셔너리"""
        pp_probs  = torch.sigmoid(outputs["pp"])[0]
        ps_probs  = torch.softmax(outputs["ps"],   dim=1)[0]
        trim_probs = torch.softmax(outputs["trim"], dim=1)[0]

        pp_labels = [PP_CLASSES[i] for i, v in enumerate(pp_probs) if v > PP_THRESHOLD]
        if not pp_labels:
            pp_labels = ["none"]

        return {
            "pattern_position": pp_labels,
            "pattern_size":     PS_CLASSES[int(ps_probs.argmax())],
            "trim":             TRIM_CLASSES[int(trim_probs.argmax())],
            "confidence": {
                "pp":   round(float(pp_probs.max()), 4),
                "ps":   round(float(ps_probs.max()),  4),
                "trim": round(float(trim_probs.max()), 4),
            }
        }

    def predict(self, source: Union[str, Image.Image]) -> dict:
        """
        단일 이미지 추론
        source: URL 문자열 또는 PIL Image
        returns: {
            "pattern_position": ["allover"],
            "pattern_size":     "medium",
            "trim":             "banded",
            "confidence":       {"pp": 0.92, "ps": 0.87, "trim": 0.79}
        }
        """
        image = self._load_image(source)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        return self._postprocess(outputs)

    def predict_url(self, url: str) -> dict:
        """URL로 추론 (predict의 alias)"""
        return self.predict(url)

    def predict_batch(self, sources: list, batch_size: int = 32) -> list[dict]:
        """
        배치 추론
        sources: URL 리스트 또는 PIL Image 리스트
        returns: 결과 딕셔너리 리스트
        """
        results = []

        for start in range(0, len(sources), batch_size):
            batch = sources[start:start + batch_size]
            images, valid_idx = [], []

            for i, src in enumerate(batch):
                try:
                    img = self._load_image(src)
                    images.append(self.preprocess(img))
                    valid_idx.append(i)
                except Exception as e:
                    print(f"[Warn] 로드 실패: {e}")
                    results.append(None)

            if not images:
                continue

            tensor = torch.stack(images).to(self.device)
            with torch.no_grad():
                outputs = {k: v for k, v in self.model(tensor).items()}

            for i in range(len(images)):
                single_output = {
                    "pp":   outputs["pp"][i:i+1],
                    "ps":   outputs["ps"][i:i+1],
                    "trim": outputs["trim"][i:i+1],
                }
                results.append(self._postprocess(single_output))

            print(f"[Batch] {min(start + batch_size, len(sources))}/{len(sources)} 완료", end="\r")

        print()
        return results


# ── 단독 실행 테스트 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    MODEL_PATH = r"C:\Users\USER\Downloads\FashionCLIP\fashion_classifier_v4.pt"

    clf = FashionClassifier(MODEL_PATH)

    # 테스트 URL (실제 R2 이미지 URL로 교체)
    test_url = "https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev/image/street/869118.jpg"

    print("\n── 단일 추론 테스트 ──")
    result = clf.predict_url(test_url)
    print(f"  pattern_position: {result['pattern_position']}")
    print(f"  pattern_size:     {result['pattern_size']}")
    print(f"  trim:             {result['trim']}")
    print(f"  confidence:       {result['confidence']}")

    print("\n── 배치 추론 테스트 ──")
    urls = [test_url] * 3
    results = clf.predict_batch(urls)
    for i, r in enumerate(results):
        print(f"  [{i}] pp={r['pattern_position']} ps={r['pattern_size']} trim={r['trim']}")