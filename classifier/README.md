# K-Fashion 속성 분류기

marqo-fashionSigLIP 백본 기반 패션 속성 분류기입니다.

## 모델 구성

| 모델 파일 | 대상 | 속성 |
|-----------|------|------|
| `fashion_classifier_v4.pt` | 전체 의류 | pattern_position / pattern_size / trim |
| `fashion_classifier_bottom.pt` | 하의 전용 | bottom_length / bottom_waist_rise |

## 분류 속성

**v4 (전체 의류)**

| 속성 | 클래스 | 비고 |
|------|--------|------|
| Pattern Position | none / allover / front / back / hem / upper / side / sleeve | 멀티셀렉트 |
| Pattern Size | none / tiny / medium / large | |
| Trim | plain / banded / mixed / rolled / unknown | standard+raw → plain 병합 |

**bottom (하의 전용)**

| 속성 | 클래스 |
|------|--------|
| Bottom Length | 발목 / 미디 / 초숏 / 숏 / 카프리 / 맥시 / 버뮤다 / 판별불가 |
| Bottom Waist Rise | 하이웨이스트 / normal / 판별불가 |

## 성능

| 분류기 | 지표 | 값 |
|--------|------|----|
| Pattern Position | micro F1 | **0.79** |
| Pattern Size | accuracy | **80%** |
| Trim | accuracy | **81.7%** |
| Bottom Length | accuracy | **73%** |
| Bottom Waist Rise | accuracy | **77%** |

## 파일 구조

```
classifier/
├── train_classifier_v4.py       # v4 학습 코드
├── train_bottom_classifier.py   # bottom 학습 코드
├── inference.py                 # 추론 모듈
├── fashion_classifier_v4.pt     # v4 모델 가중치
├── fashion_classifier_bottom.pt # bottom 모델 가중치
└── README.md
```

## 사용법

```python
from classifier.inference import FashionClassifier

clf = FashionClassifier(
    model_path        = "classifier/fashion_classifier_v4.pt",
    bottom_model_path = "classifier/fashion_classifier_bottom.pt",
)

# 상의 / 아우터 / 원피스
result = clf.predict_url("https://...", is_bottom=False)
# {
#   "pattern_position": ["allover"],
#   "pattern_size":     "medium",
#   "trim":             "banded",
#   "confidence":       {"pp": 0.92, "ps": 0.87, "trim": 0.79}
# }

# 하의
result = clf.predict_url("https://...", is_bottom=True)
# {
#   "pattern_position":  ["none"],
#   "pattern_size":      "none",
#   "trim":              "plain",
#   "confidence":        {...},
#   "bottom_length":     "미디",
#   "bottom_waist_rise": "하이웨이스트",
#   "confidence_bottom": {"length": 0.85, "waist": 0.91}
# }

# 배치 (이미지별 하의 여부 지정)
results = clf.predict_batch(
    ["url1", "url2", "url3"],
    is_bottom=[False, True, False],
)
```

## 학습 데이터

**v4**
- 총 4,241장 (project-19 / 21 / 24 / 25)
- project-24: 기존 어노테이터(ID 1,4,5,6)만 사용
- 마스킹 이미지(배경 제거) 활용

**bottom**
- 총 750장 (project-26)
- 어노테이터 ID 1, 4

## 환경

```
Python 3.11
torch (CUDA 12.1)
open-clip-torch
```