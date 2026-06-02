"""
caption.py — Gemini 패션 이미지 캡션 생성
- response_schema로 JSON 구조 강제 (automaton 기반 constrained decoding)
- asyncio + Semaphore 비동기 병렬 처리 (10만 건 대응)
- JSONL 체크포인트: 중단 후 재시작 시 완료분 자동 스킵

VLM 출력 (3 필드):
  category      — 서브카테고리 ("스트레이트 데님팬츠")
  micro_details — 기존 taxonomy 미커버 디테일 (워싱, 텍스처, 섬세한 부자재)
  mood_and_tpo  — 검색용 TPO 태그 3~5개

인덱스 필드 (빌드 함수 제공):
  dense_caption — E5 임베딩용 (기존 레이블 + VLM 출력 전체)
  flat_tags     — BM25용 (동일 토큰, 공백 구분)
"""
import asyncio
import io
import json
import re
import time
from pathlib import Path
from typing_extensions import TypedDict

import requests
from PIL import Image
from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_MODEL

CONCURRENCY = 70  # 동시 API 호출 수 (Gemini Flash ~2000 RPM 기준)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "caption_system_prompt.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()

# BM25에서 검색 의미가 없는 값 제외
_SKIP_TOKENS = {"무지", ""}

_TYPE_TO_SUFFIX = {"상의": "top", "하의": "bottom", "아우터": "outerwear", "원피스": "dress"}


class CaptionOutput(TypedDict):
    category: str             # 서브카테고리: 스트레이트 데님팬츠 / 트위드 블레이저
    micro_details: list[str]  # 기존 taxonomy 미커버: 생지, 골지, 금장 버튼, 핀턱...
    mood_and_tpo: list[str]   # TPO 태그: 하객룩, 이지웨어, 바캉스룩


def _collect_tokens(vlm: "CaptionOutput", existing: dict | None) -> list[str]:
    """dense_caption / flat_tags 공통 토큰 목록을 순서대로 반환 (중복 제거)."""
    tokens: list[str] = []
    seen: set[str] = set()

    def add(t: str) -> None:
        t = t.strip()
        if t and t not in _SKIP_TOKENS and t not in seen:
            seen.add(t)
            tokens.append(t)

    # 1) VLM 서브카테고리
    if vlm.get("category"):
        add(vlm["category"])

    if existing:
        for key in ("핏", "색상", "서브색상"):
            if existing.get(key):
                add(existing[key])
        for mat in (existing.get("소재") or []):
            add(mat)
        if existing.get("기장"):
            add(existing["기장"] + "기장")
        if existing.get("소매기장"):
            add(existing["소매기장"])
        if existing.get("넥라인"):
            add(existing["넥라인"])
        for p in (existing.get("프린트") or []):
            add(p)  # "무지"는 _SKIP_TOKENS에 포함되어 자동 제외
        for d in (existing.get("디테일") or []):
            add(d)

    # 2) VLM micro_details — 기존 디테일과 정확히 같은 토큰은 seen으로 자동 제외
    for m in (vlm.get("micro_details") or []):
        add(m)

    # 3) TPO 태그
    for t in (vlm.get("mood_and_tpo") or []):
        add(t)

    return tokens


def build_dense_caption(vlm: "CaptionOutput", existing: dict | None = None) -> str:
    """E5 임베딩용 dense_caption 문자열 (콤마 구분)."""
    return ", ".join(_collect_tokens(vlm, existing))


def build_flat_tags(vlm: "CaptionOutput", existing: dict | None = None) -> str:
    """BM25 역색인용 flat_tags 문자열 (공백 구분, 붙여쓰기 정규화)."""
    tokens = _collect_tokens(vlm, existing)
    # 공백이 포함된 토큰은 붙여쓰기로 변환 ("화이트 스티치" → "화이트스티치")
    normalized = [t.replace(" ", "") for t in tokens]
    return " ".join(normalized)


# ── Gemini 클라이언트 ──────────────────────────────────────────────────────────

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _make_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=CaptionOutput,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


def _fetch_image_bytes(url: str) -> tuple[bytes, str]:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    mime = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
    return resp.content, mime


def _pil_to_bytes(image: Image.Image) -> tuple[bytes, str]:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue(), "image/jpeg"


def _parse_item_info(item_info: str) -> dict:
    """task['data']['item_info'] 텍스트를 구조화 dict로 파싱.

    item_info 형식:
        타입: 상의 / 티셔츠
        컬러: 레드 / 화이트
        핏: 노멀
        기장: -
        소재: 저지
        디테일: -
        프린트: 스트라이프
    """
    result: dict = {}
    for line in (item_info or "").strip().splitlines():
        if ": " not in line:
            continue
        key, _, val = line.partition(": ")
        key, val = key.strip(), val.strip()
        if not val or val == "-":
            continue

        if key == "타입":
            parts = [p.strip() for p in val.split("/")]
            if len(parts) >= 2:
                result["카테고리"] = parts[1]  # 서브카테고리 (티셔츠, 팬츠 등)
            elif parts:
                result["카테고리"] = parts[0]
        elif key == "컬러":
            parts = [p.strip() for p in val.split("/")]
            result["색상"] = parts[0]
            if len(parts) >= 2 and parts[1]:
                result["서브색상"] = parts[1]
        elif key in ("소재", "디테일", "프린트"):
            result[key] = [v.strip() for v in re.split(r"[/,]", val) if v.strip()]
        elif key in ("핏", "기장", "소매기장", "넥라인"):
            result[key] = val
    return result


def _parse_retry_delay(err_str: str) -> int:
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)", err_str)
    return int(m.group(1)) + 2 if m else 60


# ── 동기 API (단건 테스트용) ─────────────────────────────────────────────────

def generate_caption(image_url: str, retries: int = 3) -> CaptionOutput:
    """R2 URL → 캡션 (동기, 단건)"""
    img_bytes, mime_type = _fetch_image_bytes(image_url)
    return _call_gemini_sync(img_bytes, mime_type, retries)


def generate_caption_from_image(image: Image.Image, retries: int = 3) -> CaptionOutput:
    """폴리곤 마스킹된 PIL Image → 캡션 (동기, 단건)"""
    img_bytes, mime_type = _pil_to_bytes(image)
    return _call_gemini_sync(img_bytes, mime_type, retries)


_TPO_NORMALIZE = {"스트릿": "스트리트", "streetwear": "스트리트"}
_ALLOWED_TPO = {
    "하객룩","오피스룩","데이트룩","피크닉룩","바캉스룩","페스티벌룩",
    "등산룩","골프룩","리조트룩","웨딩게스트","미니멀룩","고프코어",
    "Y2K","뉴트로","페미닌","클래식","스트리트","이지웨어","데일리룩","캐주얼",
}

def _postprocess(output: dict) -> CaptionOutput:
    """VLM 출력 후처리: 규칙 위반 보정."""
    tpo = output.get("mood_and_tpo") or []
    # 표기 정규화 후 허용 태그풀 외 제거
    tpo = [_TPO_NORMALIZE.get(t, t) for t in tpo]
    tpo = [t for t in tpo if t in _ALLOWED_TPO]
    # 데일리룩+캐주얼 동시 출현 금지
    if "데일리룩" in tpo and "캐주얼" in tpo:
        tpo = [t for t in tpo if t != "캐주얼"]
    output["mood_and_tpo"] = tpo
    return output


def _call_gemini_sync(img_bytes: bytes, mime_type: str, retries: int) -> CaptionOutput:
    client = _get_client()
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                    SYSTEM_PROMPT,
                ],
                config=_make_config(),
            )
            return _postprocess(json.loads(response.text))
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                wait = _parse_retry_delay(err_str)
                if attempt < retries - 1:
                    print(f"  rate limit — {wait}s 대기 ({attempt+1}/{retries})")
                    time.sleep(wait)
                else:
                    raise
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


# ── 비동기 API (배치 처리) ───────────────────────────────────────────────────

async def _fetch_image_bytes_async(url: str) -> tuple[bytes, str]:
    return await asyncio.to_thread(_fetch_image_bytes, url)


async def _call_gemini_async(
    img_bytes: bytes,
    mime_type: str,
    retries: int = 4,
) -> CaptionOutput:
    client = _get_client()
    for attempt in range(retries):
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                        SYSTEM_PROMPT,
                    ],
                    config=_make_config(),
                ),
                timeout=120,
            )
            return _postprocess(json.loads(response.text))
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if attempt < retries - 1:
                wait = _parse_retry_delay(err_str) if is_rate_limit else 2 ** attempt
                await asyncio.sleep(wait)
            else:
                raise


def _local_image_path(task: dict, masked_dir: Path | None) -> Path | None:
    """task → 로컬 masked image 경로 반환. 없으면 None."""
    if masked_dir is None:
        return None
    file_id = task["data"].get("file_id", "")
    item_info = task["data"].get("item_info", "")
    kor_type = ""
    for line in item_info.strip().splitlines():
        if line.startswith("타입:"):
            kor_type = line.split(":", 1)[1].strip().split("/")[0].strip()
            break
    suffix = _TYPE_TO_SUFFIX.get(kor_type)
    if not suffix:
        return None
    p = masked_dir / f"{file_id}_{suffix}.jpg"
    return p if p.exists() else None


# ── 이미지 디렉토리 직접 배치 처리 ──────────────────────────────────────────

async def _process_image(
    img_path: Path,
    sem: asyncio.Semaphore,
    out_path: Path,
    counters: dict,
    write_lock: asyncio.Lock,
) -> None:
    # 파일명: {file_id}_{category}.jpg
    stem = img_path.stem
    parts = stem.rsplit("_", 1)
    file_id = parts[0]
    category = parts[1] if len(parts) == 2 else ""

    async with sem:
        try:
            img_bytes = await asyncio.to_thread(img_path.read_bytes)
            vlm = await _call_gemini_async(img_bytes, "image/jpeg")

            row = {
                "file_id": file_id,
                "category": category,
                "caption_category":      vlm.get("category", ""),
                "caption_micro_details": vlm.get("micro_details", []),
                "mood_and_tpo":          vlm.get("mood_and_tpo", []),
            }
            counters["ok"] += 1
            status = "OK "
        except Exception as e:
            row = {
                "file_id": file_id,
                "category": category,
                "caption_category": "",
                "caption_micro_details": [],
                "mood_and_tpo": [],
                "error": str(e),
            }
            counters["err"] += 1
            err_short = str(e)[:80]
            status = f"ERR [{err_short}]"

        async with write_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        counters["done"] += 1
        total = counters["total"]
        elapsed = time.time() - counters["start"]
        rps = counters["done"] / elapsed if elapsed > 0 else 0
        eta = (total - counters["done"]) / rps if rps > 0 else 0
        ts = time.strftime("%H:%M:%S")
        print(
            f"[{ts}] [{counters['done']:>6}/{total}] {status}  {img_path.name:<35} "
            f"err={counters['err']}  {rps:.1f}req/s  ETA={eta/60:.1f}min"
        )


async def batch_from_dir_async(
    images_dir: str,
    out_path: str,
    limit: int | None = None,
    concurrency: int = CONCURRENCY,
) -> None:
    """
    masked images 디렉토리 → 비동기 병렬 캡션 생성 → JSONL 저장.
    task JSON 불필요. 파일명 {file_id}_{category}.jpg 에서 메타데이터 추출.
    체크포인트: out_path가 이미 존재하면 완료된 (file_id, category) 쌍을 스킵.
    """
    all_images = sorted(Path(images_dir).glob("*.jpg"))
    if limit:
        all_images = all_images[:limit]

    out_p = Path(out_path)
    done_keys: set[tuple] = set()
    if out_p.exists():
        with open(out_p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("error"):  # 성공한 것만 스킵
                        done_keys.add((r["file_id"], r["category"]))
                except Exception:
                    pass
        if done_keys:
            print(f"체크포인트 감지: {len(done_keys)}개 완료, 이어서 처리")

    def _key(p: Path):
        parts = p.stem.rsplit("_", 1)
        return (parts[0], parts[1] if len(parts) == 2 else "")

    remaining = [p for p in all_images if _key(p) not in done_keys]

    if not remaining:
        print("모두 완료됨.")
        return

    print(f"처리 대상: {len(remaining)}개 / 전체 {len(all_images)}개  (동시성: {concurrency})")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    queue      = asyncio.Queue()
    write_lock = asyncio.Lock()
    sem        = asyncio.Semaphore(concurrency)
    counters   = {"ok": 0, "err": 0, "done": 0, "total": len(remaining), "start": time.time()}

    for p in remaining:
        queue.put_nowait(p)

    async def worker():
        while True:
            try:
                img_path = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await _process_image(img_path, sem, out_p, counters, write_lock)

    await asyncio.gather(*[worker() for _ in range(concurrency)])

    elapsed = time.time() - counters["start"]
    print(f"\n완료: {counters['ok']}개 성공, {counters['err']}개 실패  ({elapsed/60:.1f}분)")
    print(f"→ {out_path}")


def batch_from_dir(
    images_dir: str,
    out_path: str,
    limit: int | None = None,
    concurrency: int = CONCURRENCY,
) -> None:
    """batch_from_dir_async의 동기 진입점."""
    asyncio.run(batch_from_dir_async(images_dir, out_path, limit=limit, concurrency=concurrency))


async def _process_task(
    task: dict,
    sem: asyncio.Semaphore,
    out_path: Path,
    counters: dict,
    write_lock: asyncio.Lock,
    masked_dir: Path | None = None,
) -> None:
    url = task["data"]["image"]

    local_path = _local_image_path(task, masked_dir)
    if local_path:
        fname = local_path.name
    else:
        fname = url.split("/")[-1]

    async with sem:
        try:
            if local_path:
                img_bytes = local_path.read_bytes()
                mime_type = "image/jpeg"
            else:
                img_bytes, mime_type = await _fetch_image_bytes_async(url)
            vlm = await _call_gemini_async(img_bytes, mime_type)

            # VLM 출력 저장 (기존 레이블과 분리; 인덱스 결합은 인덱서에서)
            task["data"]["caption_category"]      = vlm.get("category", "")
            task["data"]["caption_micro_details"] = vlm.get("micro_details", [])
            task["data"]["mood_and_tpo"]          = vlm.get("mood_and_tpo", [])

            counters["ok"] += 1
            status = "OK "
        except Exception as e:
            task["data"]["caption_category"]      = ""
            task["data"]["caption_micro_details"] = []
            task["data"]["mood_and_tpo"]          = []
            counters["err"] += 1
            status = f"ERR {e}"

        async with write_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

        counters["done"] += 1
        total = counters["total"]
        elapsed = time.time() - counters["start"]
        rps = counters["done"] / elapsed if elapsed > 0 else 0
        eta = (total - counters["done"]) / rps if rps > 0 else 0
        print(
            f"[{counters['done']:>6}/{total}] {status:<6} {fname:<30} "
            f"err={counters['err']}  {rps:.1f}req/s  ETA={eta/60:.1f}min"
        )


async def batch_generate_async(
    tasks_path: str,
    out_path: str,
    limit: int | None = None,
    concurrency: int = CONCURRENCY,
    masked_images_dir: str | None = None,
) -> None:
    """
    tasks JSON → 비동기 병렬 캡션 생성 → JSONL 저장.
    masked_images_dir: 로컬 masked image 디렉토리 경로. 지정 시 R2 대신 로컬 파일 우선 사용.
    체크포인트: out_path가 이미 존재하면 완료된 file_id를 스킵하고 이어서 처리.
    """
    with open(tasks_path, encoding="utf-8") as f:
        tasks = json.load(f)

    if limit:
        tasks = tasks[:limit]

    out_p = Path(out_path)
    done_ids: set = set()
    if out_p.exists():
        with open(out_p, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["data"]["file_id"])
                except Exception:
                    pass
        if done_ids:
            print(f"체크포인트 감지: {len(done_ids)}개 완료, 이어서 처리")

    remaining = [t for t in tasks if t["data"]["file_id"] not in done_ids]

    if not remaining:
        print("모두 완료됨.")
        return

    masked_dir = Path(masked_images_dir) if masked_images_dir else None
    src_label = f"로컬({masked_dir.name})" if masked_dir else "R2 URL"
    print(f"처리 대상: {len(remaining)}개 / 전체 {len(tasks)}개  (동시성: {concurrency}, 이미지 소스: {src_label})")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    sem        = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counters   = {"ok": 0, "err": 0, "done": 0, "total": len(remaining), "start": time.time()}

    await asyncio.gather(*[
        _process_task(task, sem, out_p, counters, write_lock, masked_dir=masked_dir)
        for task in remaining
    ])

    elapsed = time.time() - counters["start"]
    print(f"\n완료: {counters['ok']}개 성공, {counters['err']}개 실패  ({elapsed/60:.1f}분)")
    print(f"→ {out_path}")


def batch_generate(
    tasks_path: str,
    out_path: str,
    limit: int | None = None,
    concurrency: int = CONCURRENCY,
    masked_images_dir: str | None = None,
) -> None:
    """batch_generate_async의 동기 진입점."""
    asyncio.run(batch_generate_async(
        tasks_path, out_path,
        limit=limit, concurrency=concurrency,
        masked_images_dir=masked_images_dir,
    ))
