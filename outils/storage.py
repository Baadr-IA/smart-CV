import json
import logging
import os
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from .fs_utils import sanitize_client_filename

logger = logging.getLogger("storage")


def _normalize_prefix(prefix: str) -> str:
    value = (prefix or "").strip()
    if not value:
        return ""
    value = value.replace("\\", "/")
    if not value.endswith("/"):
        value += "/"
    return value.lstrip("/")


def storage_mode() -> str:
    return os.getenv("STORAGE_MODE", "local").strip().lower()


def s3_enabled() -> bool:
    return storage_mode() == "s3"


def s3_bucket() -> Optional[str]:
    return os.getenv("S3_BUCKET")


def s3_region() -> Optional[str]:
    return os.getenv("S3_REGION") or os.getenv("AWS_REGION")


def s3_input_prefix() -> str:
    return _normalize_prefix(os.getenv("S3_INPUT_PREFIX", "input/"))


def s3_output_prefix() -> str:
    return _normalize_prefix(os.getenv("S3_OUTPUT_PREFIX", "output/"))


def s3_key(prefix: str, filename: str) -> str:
    safe = sanitize_client_filename(filename)
    return f"{_normalize_prefix(prefix)}{safe}"


def get_s3_client():
    region = s3_region()
    if region:
        return boto3.client("s3", region_name=region)
    return boto3.client("s3")


def download_to_path(bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = get_s3_client()
    try:
        client.download_file(bucket, key, str(dest))
    except ClientError as exc:
        logger.error("S3 download failed: s3://%s/%s (%s)", bucket, key, exc)
        raise


def upload_file(bucket: str, key: str, path: Path, content_type: Optional[str] = None) -> None:
    client = get_s3_client()
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type
    try:
        if extra_args:
            client.upload_file(str(path), bucket, key, ExtraArgs=extra_args)
        else:
            client.upload_file(str(path), bucket, key)
    except ClientError as exc:
        logger.error("S3 upload failed: s3://%s/%s (%s)", bucket, key, exc)
        raise


def upload_json(bucket: str, key: str, payload: dict) -> None:
    client = get_s3_client()
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json; charset=utf-8",
        )
    except ClientError as exc:
        logger.error("S3 put_object failed: s3://%s/%s (%s)", bucket, key, exc)
        raise


def download_json(bucket: str, key: str) -> dict:
    client = get_s3_client()
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read()
        return json.loads(body)
    except ClientError as exc:
        logger.error("S3 get_object failed: s3://%s/%s (%s)", bucket, key, exc)
        raise
