# K-Fashion 속성 분류기 (v4)

marqo-fashionSigLIP 백본 기반 패션 속성 분류기입니다.

## 분류 속성

| 속성 | 클래스 | 비고 |
|------|--------|------|
| Pattern Position | none / allover / front / back / hem / upper / side / sleeve | 멀티셀렉트 |
| Pattern Size | none / tiny / medium / large | |
| Trim | plain / banded / mixed / rolled / unknown | standard+raw → plain 병합 |

## 성능 (v4)

| 분류기 | 지표 | 값 |
|--------|------|----|
| Pattern Position | micro F1 | 0.80 |
| Pattern Size | accuracy | 83% |
| Trim | accuracy | 77% |

## 파일 구조

```
classifier/
├── train.py                     # 학습 코드
├── inference.py                 # 추론 모듈
├── extract_samples.py           # 희귀 클래스 샘플 추출
├── low_confidence.py            # confidence 낮은 샘플 추출
└── fashion_classifier_v4.pt     # 학습된 모델 가중치
```

## 사용법

```python
from classifier.inference import FashionClassifier

clf = FashionClassifier("classifier/fashion_classifier_v4.pt")

# 단일 이미지
result = clf.predict_url("https://...")
print(result)
# {
#   "pattern_position": ["allover"],
#   "pattern_size": "medium",
#   "trim": "banded",
#   "confidence": {"pp": 0.92, "ps": 0.87, "trim": 0.79}
# }

# 배치
results = clf.predict_batch(["url1", "url2", ...])
```

## 학습 데이터

- 총 4760장
- 마스킹 이미지(배경 제거) 활용

## 환경

```
Python 3.11
torch (CUDA 12.1)
open-clip-torch
```
