import os

import requests

from src.config import LS_URL, LS_API_TOKEN, LS_PROJECT_ID

def _auth_header() -> str:
    """LS 1.x: Legacy Token → 'Token …', 일부 배포본은 Bearer PAT."""
    scheme = os.environ.get("LS_AUTH_SCHEME", "Token").strip()
    return f"{scheme} {LS_API_TOKEN}"


HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
}


def _headers() -> dict:
    return {**HEADERS, "Authorization": _auth_header()}


def get_project():
    return requests.get(f"{LS_URL}/api/projects/{LS_PROJECT_ID}", headers=_headers())


def get_all_tasks() -> dict:
    """file_id → ls_task_id 매핑 반환"""
    tasks_map = {}
    page = 1
    while True:
        resp = requests.get(
            f"{LS_URL}/api/tasks",
            headers=_headers(),
            params={"project": LS_PROJECT_ID, "page": page, "page_size": 100},
        )
        if resp.status_code != 200:
            print(f"조회 실패: {resp.status_code}")
            break
        data = resp.json()
        task_list = data.get("tasks", [])
        if not task_list:
            break
        for t in task_list:
            fid = t.get("data", {}).get("file_id")
            if fid:
                tasks_map[int(fid)] = t["id"]
        if not data.get("next"):
            break
        page += 1
    return tasks_map


def patch_task(ls_task_id: int, data: dict) -> requests.Response:
    return requests.patch(
        f"{LS_URL}/api/tasks/{ls_task_id}",
        headers=_headers(),
        json={"data": data},
    )


def get_task(ls_task_id: int) -> requests.Response:
    return requests.get(f"{LS_URL}/api/tasks/{ls_task_id}", headers=_headers())


def _require_ls_config() -> None:
    if not LS_URL or not LS_API_TOKEN or not LS_PROJECT_ID:
        raise RuntimeError(
            "Label Studio 설정이 없습니다. 프로젝트 루트 .env 에 "
            "LS_URL, LS_API_TOKEN, LS_PROJECT_ID 를 설정하세요."
        )


def import_tasks(tasks: list[dict], *, return_task_ids: bool = False) -> dict:
    """
    Label Studio 프로젝트에 task 일괄 등록.
    POST /api/projects/{id}/import
    """
    _require_ls_config()
    resp = requests.post(
        f"{LS_URL}/api/projects/{LS_PROJECT_ID}/import",
        headers=_headers(),
        json=tasks,
        timeout=120,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"import 실패 {resp.status_code}: {resp.text[:500]}")

    body = resp.json() if resp.content else {}
    if return_task_ids:
        return body
    return {"status": resp.status_code, "count": len(tasks), "response": body}
