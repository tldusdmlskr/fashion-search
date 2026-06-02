"""
test_auth.py — Label Studio 인증 방법 테스트
실행: python scripts/test_auth.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os, requests

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
token  = os.environ.get("LS_API_TOKEN", "")
url    = os.environ.get("LS_URL", "")
pid    = os.environ.get("LS_PROJECT_ID", "")
email  = os.environ.get("LS_EMAIL", "")
passwd = os.environ.get("LS_PASSWORD", "")

print("=== 방법 1: Bearer 토큰 ===")
r = requests.get(f"{url}/api/current-user/whoami",
                 headers={"Authorization": f"Bearer {token}", "ngrok-skip-browser-warning": "true"}, timeout=5)
print(f"  {r.status_code}: {r.text[:120]}")

print("\n=== 방법 2: 세션 로그인 (이메일/비밀번호) ===")
if email and passwd:
    s = requests.Session()
    s.headers.update({"ngrok-skip-browser-warning": "true"})
    # CSRF 토큰 취득
    s.get(f"{url}/user/login/")
    csrf = s.cookies.get("csrftoken", "")
    login_resp = s.post(f"{url}/user/login/",
                        data={"email": email, "password": passwd, "csrfmiddlewaretoken": csrf},
                        headers={"Referer": f"{url}/user/login/"})
    print(f"  로그인: {login_resp.status_code}")
    whoami = s.get(f"{url}/api/current-user/whoami")
    print(f"  whoami: {whoami.status_code}: {whoami.text[:120]}")
else:
    print("  .env에 LS_EMAIL, LS_PASSWORD 설정 필요")

print("\n=== 방법 3: 팀원에게 확인 필요한 사항 ===")
print("  팀원 LS 설정 → Organization → API Token 활성화 여부")
print("  또는 팀원이 직접 본인 API 토큰을 공유해야 할 수 있음")
