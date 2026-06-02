"""R2 라벨 JSON 접근."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Iterator

import boto3
from botocore.config import Config

from r2_credentials import (
    R2_ACCESS_KEY,
    R2_ACCOUNT_ID,
    R2_BUCKET,
    R2_IMAGE_PREFIX,
    R2_LABEL_PREFIX,
    R2_SECRET_KEY,
)


@dataclass
class R2Config:
    account_id: str
    access_key: str
    secret_key: str
    bucket: str
    image_prefix: str
    label_prefix: str

    @classmethod
    def from_defaults(cls) -> R2Config:
        missing = [
            name
            for name, value in [
                ("R2_ACCOUNT_ID", R2_ACCOUNT_ID),
                ("R2_ACCESS_KEY", R2_ACCESS_KEY),
                ("R2_SECRET_KEY", R2_SECRET_KEY),
            ]
            if not value
        ]
        if missing:
            raise ValueError(
                f"annotation_setting/r2_credentials.py 에 값을 채워 주세요: {', '.join(missing)}"
            )

        image_prefix = R2_IMAGE_PREFIX
        label_prefix = R2_LABEL_PREFIX
        if image_prefix and not image_prefix.endswith("/"):
            image_prefix += "/"
        if label_prefix and not label_prefix.endswith("/"):
            label_prefix += "/"

        return cls(
            account_id=R2_ACCOUNT_ID,
            access_key=R2_ACCESS_KEY,
            secret_key=R2_SECRET_KEY,
            bucket=R2_BUCKET,
            image_prefix=image_prefix,
            label_prefix=label_prefix,
        )


def create_r2_client(config: R2Config):
    return boto3.client(
        service_name="s3",
        endpoint_url=f"https://{config.account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        config=Config(signature_version="s3v4"),
    )


def iter_label_keys(s3_client, config: R2Config) -> Iterator[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=config.bucket, Prefix=config.label_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                yield key


def load_json_from_r2(s3_client, config: R2Config, key: str) -> dict[str, Any]:
    response = s3_client.get_object(Bucket=config.bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


_THREAD_LOCAL = threading.local()


def get_thread_client(config: R2Config):
    cached = getattr(_THREAD_LOCAL, "config", None)
    if cached != config or not hasattr(_THREAD_LOCAL, "s3_client"):
        _THREAD_LOCAL.config = config
        _THREAD_LOCAL.s3_client = create_r2_client(config)
    return _THREAD_LOCAL.s3_client
