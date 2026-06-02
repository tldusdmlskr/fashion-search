from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from libsql import connect

ROOT = Path(__file__).parent.parent
ENV_PATH = ROOT / ".env"

CATEGORY_TO_CLOTHING_TYPE = {
    "top": "상의",
    "bottom": "하의",
    "outerwear": "아우터",
    "dress": "원피스",
}

DEFAULT_CAPTION_PATHS = [
    ROOT / "output" / "captions_full_lite.jsonl",
    ROOT / "output" / "captions_full.jsonl",
]

BATCH_SIZE = 200          # 500 → 200: 배치당 read 부담 감소
BATCH_SLEEP = 0.3         # 배치 사이 sleep(초): burst 방지
STAGING_TABLE = "staging_caption_annotations"
ANNOTATIONS_TABLE = "caption_annotations"
CHECKPOINT_PATH = ROOT / ".inject_captions.checkpoint.json"


def connect_db() -> connect:
    load_dotenv(ENV_PATH)
    db_url = os.environ.get("DB_URL")
    db_token = os.environ.get("DB_ACCESS_TOKEN")
    if not db_url or not db_token:
        raise RuntimeError("DB_URL and DB_ACCESS_TOKEN must be set in .env")

    print("Connecting to DB...", flush=True)
    conn = connect(db_url, auth_token=db_token, _uri=True)
    print("DB connected", flush=True)
    return conn


def init_tables(cursor) -> None:
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {STAGING_TABLE} (
            image_id INTEGER NOT NULL,
            clothing_type TEXT NOT NULL,
            caption_category TEXT,
            caption_micro_details TEXT,
            mood_and_tpo TEXT,
            PRIMARY KEY(image_id, clothing_type)
        )
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ANNOTATIONS_TABLE} (
            image_id INTEGER NOT NULL,
            clothing_type TEXT NOT NULL,
            caption_category TEXT,
            caption_micro_details TEXT,
            mood_and_tpo TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(image_id, clothing_type)
        )
        """
    )


def clear_staging(cursor) -> None:
    print("Clearing staging table...", flush=True)
    cursor.execute(f"DELETE FROM {STAGING_TABLE}")


def load_checkpoint(paths: list[Path]) -> dict | None:
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    expected_files = [str(path.resolve()) for path in paths]
    if checkpoint.get("files") != expected_files:
        return None
    if checkpoint.get("phase") == "finished":
        return None
    return checkpoint


def save_checkpoint(checkpoint: dict) -> None:
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")


def clear_checkpoint() -> None:
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


def initialize_checkpoint(paths: list[Path]) -> dict:
    checkpoint = {
        "files": [str(path.resolve()) for path in paths],
        "current_index": 0,
        "current_line": 1,
        "phase": "insert",
    }
    save_checkpoint(checkpoint)
    return checkpoint


def update_checkpoint(checkpoint: dict, current_index: int | None = None, current_line: int | None = None, phase: str | None = None) -> None:
    if current_index is not None:
        checkpoint["current_index"] = current_index
    if current_line is not None:
        checkpoint["current_line"] = current_line
    if phase is not None:
        checkpoint["phase"] = phase
    save_checkpoint(checkpoint)


def process_caption_file(cursor, conn, path: Path, start_line: int, checkpoint: dict, file_index: int) -> int:
    rows: list[tuple] = []
    inserted = 0
    sql = (
        f"INSERT OR REPLACE INTO {STAGING_TABLE} "
        "(image_id, clothing_type, caption_category, caption_micro_details, mood_and_tpo) "
        "VALUES (?, ?, ?, ?, ?)"
    )

    print(f"Processing {path} from line {start_line}...", flush=True)
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no < start_line:
                continue

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc

            file_id = str(record.get("file_id") or record.get("image_id"))
            if not file_id:
                raise ValueError(f"Missing file_id in line {line_no} of {path}")

            category = record.get("category")
            clothing_type = CATEGORY_TO_CLOTHING_TYPE.get(category)
            if clothing_type is None:
                continue

            caption_category = record.get("caption_category")
            caption_micro_details = json.dumps(record.get("caption_micro_details") or [], ensure_ascii=False)
            mood_and_tpo = json.dumps(record.get("mood_and_tpo") or [], ensure_ascii=False)
            rows.append((int(file_id), clothing_type, caption_category, caption_micro_details, mood_and_tpo))

            if len(rows) >= BATCH_SIZE:
                cursor.executemany(sql, rows)
                conn.commit()
                inserted += len(rows)
                rows.clear()
                update_checkpoint(checkpoint, current_index=file_index, current_line=line_no + 1)
                print(f"Inserted {inserted} rows so far...", flush=True)
                time.sleep(BATCH_SLEEP)  # burst 방지

        if rows:
            cursor.executemany(sql, rows)
            conn.commit()
            inserted += len(rows)
            update_checkpoint(checkpoint, current_index=file_index, current_line=line_no + 1)
            print(f"Inserted {inserted} rows so far...", flush=True)

    return inserted


def sync_annotations_from_staging(cursor) -> int:
    """staging → caption_annotations 단순 복사 (full scan 1회)."""
    print("Syncing annotations from staging...", flush=True)
    cursor.execute(
        f"INSERT OR REPLACE INTO {ANNOTATIONS_TABLE} "
        f"(image_id, clothing_type, caption_category, caption_micro_details, mood_and_tpo) "
        f"SELECT image_id, clothing_type, caption_category, caption_micro_details, mood_and_tpo "
        f"FROM {STAGING_TABLE}"
    )
    return cursor.rowcount


def sync_clothing_items(cursor, conn) -> int:
    """
    caption_annotations → clothing_items 업데이트.

    기존 correlated subquery 4개(×160k행 = 457억 reads) →
    Python에서 배치 UPDATE로 교체 (~160k reads 수준).
    """
    print("Fetching annotation keys...", flush=True)
    cursor.execute(
        f"SELECT image_id, clothing_type, caption_category, caption_micro_details, mood_and_tpo "
        f"FROM {ANNOTATIONS_TABLE}"
    )
    rows = cursor.fetchall()
    total = len(rows)
    print(f"Updating {total} clothing_items rows in batches...", flush=True)

    sql = (
        "UPDATE clothing_items "
        "SET caption_category = ?, caption_micro_details = ?, mood_and_tpo = ? "
        "WHERE image_id = ? AND clothing_type = ?"
    )

    updated = 0
    batch = []
    for image_id, clothing_type, caption_category, caption_micro_details, mood_and_tpo in rows:
        batch.append((caption_category, caption_micro_details, mood_and_tpo, image_id, clothing_type))

        if len(batch) >= BATCH_SIZE:
            cursor.executemany(sql, batch)
            conn.commit()
            updated += len(batch)
            batch.clear()
            print(f"Updated {updated}/{total} rows...", flush=True)
            time.sleep(BATCH_SLEEP)

    if batch:
        cursor.executemany(sql, batch)
        conn.commit()
        updated += len(batch)
        print(f"Updated {updated}/{total} rows...", flush=True)

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load JSONL captions into caption_annotations, then sync clothing_items in bulk"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="JSONL caption files to load",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not commit updates to the database")
    parser.add_argument("--reset", action="store_true", help="Discard any existing checkpoint and start from scratch")
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        print("No caption files passed as arguments; using default files:")
        for default_path in DEFAULT_CAPTION_PATHS:
            print("  ", default_path)
        paths = DEFAULT_CAPTION_PATHS

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Caption file not found: {path}")

    conn = connect_db()
    cursor = conn.cursor()
    init_tables(cursor)

    checkpoint = None if args.reset else load_checkpoint(paths)
    if checkpoint is None:
        if args.reset:
            print("Reset requested: discarding previous checkpoint.", flush=True)
        else:
            print("No valid checkpoint found; starting fresh.", flush=True)
        clear_staging(cursor)
        conn.commit()
        checkpoint = initialize_checkpoint(paths)
    else:
        print(
            f"Resuming caption injection from file {checkpoint['current_index'] + 1}/{len(paths)} "
            f"starting at line {checkpoint['current_line']}.",
            flush=True,
        )

    if checkpoint["phase"] == "insert":
        inserted = 0
        for file_index in range(checkpoint["current_index"], len(paths)):
            file_path = paths[file_index]
            inserted += process_caption_file(
                cursor,
                conn,
                file_path,
                checkpoint["current_line"],
                checkpoint,
                file_index,
            )
            update_checkpoint(checkpoint, current_index=file_index + 1, current_line=1)

        print(f"Inserted {inserted} rows into staging table {STAGING_TABLE}", flush=True)
        update_checkpoint(checkpoint, phase="sync")

    if checkpoint["phase"] == "sync":
        annotations_synced = sync_annotations_from_staging(cursor)
        conn.commit()
        print(f"Synchronized {annotations_synced} rows into {ANNOTATIONS_TABLE}", flush=True)

        updated_rows = sync_clothing_items(cursor, conn)
        print(f"Updated {updated_rows} clothing_items rows from annotations", flush=True)

        update_checkpoint(checkpoint, phase="finished")
        clear_checkpoint()
        print("Checkpoint cleared. Injection complete.", flush=True)

    if args.dry_run:
        print("Dry run enabled; rolling back changes", flush=True)
        conn.rollback()

    conn.close()


if __name__ == "__main__":
    main()
