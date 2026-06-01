## Polygon Masking

이 폴더는 R2에 저장된 이미지와 라벨링(JSON) 데이터를 기반으로 폴리곤 마스킹 이미지를 생성하고 관리하기 위한 작업 공간이다.

### 주요 파일

#### `polygon_masking/mask_from_r2.py`

Cloudflare R2에서 원본 이미지와 라벨링(JSON) 데이터를 불러와 폴리곤 마스크를 생성한 뒤, 결과 이미지를 로컬 폴더에 저장한다.

#### `polygon_masking/upload_masking_to_r2.py`

로컬에 생성된 마스킹 이미지를 R2의 `masking_data` 경로로 업로드한다.

#### `polygon_masking/validate_masks.py`

R2의 라벨링(JSON) 데이터와 `masking_data` 내 생성된 마스크 이미지를 비교하여 누락 여부를 검증한다.

### 자동화 파이프라인

자동화 구현 시 핵심적으로 사용되는 파일은 다음과 같다.

1. `mask_from_r2.py`

   * R2 데이터 기반 마스크 이미지 생성

2. `upload_masking_to_r2.py`

   * 생성된 마스크 이미지를 R2에 업로드

`validate_masks.py`는 생성 및 업로드 결과를 검증하기 위한 점검 도구로 사용된다.
