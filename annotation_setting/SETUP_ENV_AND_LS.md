# .env · Label Studio 설정 가이드

## 1. `.env` 파일 위치

```
fashion-search/.env          ← 프로젝트 루트 (README와 같은 폴더)
```

이미 `.env`가 생성되어 있습니다. **Label Studio 3줄만** 채우면 됩니다.

---

## 2. `R2_PUBLIC_URL` 확인 (이미 넣어 둠)

브라우저에서 테스트:

```
https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev/image/modern/1012265.jpg
```

(스타일·file_id는 `tasks.json`에 있는 값으로 바꿔 보세요.)

- 이미지가 보이면 → 그대로 사용
- 403/404면 → Cloudflare R2 대시보드에서 **Public bucket / r2.dev URL** 확인 후 `.env` 수정

---

## 3. Label Studio 프로젝트 만들기

1. 팀 Label Studio 주소로 로그인
2. **Create Project** (또는 기존 프로젝트 사용)
3. 프로젝트 이름 예: `bottom-waist-length-stripe`
4. **Labeling Setup**은 나중에 해도 됨 (태스크는 스크립트로 넣음)

---

## 4. `LS_PROJECT_ID` 찾기

프로젝트를 연 뒤 **주소창**을 봅니다.

| URL 예시 | 넣을 값 |
|----------|---------|
| `https://서버/projects/17/data` | `17` |
| `https://서버/projects/17/settings` | `17` |

`.env`에:

```
LS_PROJECT_ID=17
```

---

## 5. `LS_URL` 찾기

주소창에서 **서버까지만** 복사합니다 ( `/projects/...` 이후는 제외).

| 전체 URL | `LS_URL` |
|----------|----------|
| `https://abc123.ngrok-free.app/projects/17/data` | `https://abc123.ngrok-free.app` |
| `http://localhost:8080/projects/1/data` | `http://localhost:8080` |

끝에 `/` 없이.

---

## 6. `LS_API_TOKEN` 발급 (Personal Access Token)

**맞습니다.** Account & Settings → **Access Token** / **Legacy Token** 에서 새로 만든 값을 씁니다.

⚠ **브라우저 개발자도구·쿠키의 JWT(refresh)는 API 토큰이 아닙니다.** `401`이 나오면 PAT를 다시 발급하세요.

1. 우측 상단 **프로필 / 계정 아이콘**
2. **Account & Settings**
3. **Access Token** (또는 **Legacy Token**) 탭
4. **Create new token** → 표시된 문자열 복사 (보통 `eyJ…` 가 아닌 짧은 hex 형태이기도 함)
5. `.env` 저장 (**Ctrl+S** — 저장 안 하면 스크립트가 빈 값으로 읽음)

```
LS_API_TOKEN=여기에_붙여넣기
```

기본 인증 헤더는 `Token <값>` 입니다. Bearer만 되는 서버면 `.env`에 `LS_AUTH_SCHEME=Bearer` 추가.

⚠ 토큰은 Git에 올리지 마세요.

---

## 7. 연결 테스트

프로젝트 루트에서:

```powershell
cd c:\Users\cjy02\Desktop\4-1\fashion-search
python scripts/test_auth.py
```

- **방법 1**이 `200`이고 이메일/사용자 정보가 보이면 성공
- `401` → 토큰 오류
- `404` → `LS_URL` 또는 `LS_PROJECT_ID` 오류

---

## 8. 다음 단계

```powershell
python annotation_setting/collect_stripe_tasks.py
python scripts/create_tasks_from_r2.py --clothing-type 하의
python scripts/upload_tasks_to_ls.py
```
