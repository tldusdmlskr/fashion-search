# 어노테이션 태스크 파이프라인

[Data-Engineering-Project2 `dev`](https://github.com/JANENAMMMM/Data-Engineering-Project2/tree/dev) 의 `src/` + `scripts/` 구조를 이 레포 루트에 맞춰 두었습니다.

## 로컬 JSON 없이 (권장)

```
R2 샘플링                    task JSON 생성              LS 업로드
─────────────────────────────────────────────────────────────────────
collect_stripe_tasks.py  →  create_tasks_from_r2.py  →  upload_tasks_to_ls.py
(annotation_setting/)       (scripts/)                  (scripts/)
         │                           │
         └ tasks.json                └ output/tasks.json
```

### 1. R2에서 대상 이미지 ID 수집

**전체 클라우드 라벨 인덱스 (권장):**

```powershell
python annotation_setting/collect_all_tasks.py
```

**하의 샘플 (드롭웨이스트 20 + 임의 30 / 스타일, 최대 1200건):**

```powershell
python annotation_setting/collect_bottom_tasks.py
# 기본: --workers 12 --json-workers 8
```

**스트라이프만·스타일당 40건 (부분 샘플):**

```powershell
python annotation_setting/collect_stripe_tasks.py
```

- 키: `annotation_setting/r2_credentials.py`
- 결과: `annotation_setting/tasks.json` (`records`: file_id, style, label_key)

### 2. R2 라벨 메타로 Label Studio task JSON 생성

```powershell
# .env 에 R2_PUBLIC_URL 필수 (이미지 URL)
python scripts/create_tasks_from_r2.py --clothing-type 하의
```

- 결과: `output/tasks.json`

### 3. Label Studio에 등록

`.env` 에 설정:

- `LS_URL`
- `LS_API_TOKEN`
- `LS_PROJECT_ID`

```powershell
python scripts/test_auth.py
python scripts/upload_tasks_to_ls.py
```

## 로컬 JSON 캐시 경로 (선택)

한 번 전체를 받아 두고 싶을 때:

```powershell
python scripts/export_metadata.py   # R2 → data/labels/
python scripts/create_tasks.py      # 샘플링 → output/tasks.json
```

## 필요한 파일

| 구분 | 경로 | 역할 |
|------|------|------|
| R2 | `annotation_setting/r2_credentials.py` | 스트라이프 수집·R2 라벨 읽기 |
| R2 | `annotation_setting/r2_io.py` | S3 클라이언트 |
| 코어 | `src/label_parser.py`, `task_builder.py`, `url_utils.py` | JSON → task dict |
| LS | `src/ls_client.py` | REST import / patch |
| 설정 | `.env` | LS URL·토큰, **R2_PUBLIC_URL** |
| 스펙 | `bottom_attributes_spec.en.json` | 하의 허리선·기장 7단계 |

## Label Studio 키

- `create_tasks*.py` → JSON만 만듦 (키 불필요)
- `upload_tasks_to_ls.py` → **API 토큰·프로젝트 ID 필수**

## 현재 `annotation_setting/create_tasks.py`

루트 `scripts/create_tasks.py` 와 동일한 **로컬 `data/labels/`** 용 스크립트입니다.  
클라우드만 쓸 때는 **`scripts/create_tasks_from_r2.py`** 를 사용하세요.
