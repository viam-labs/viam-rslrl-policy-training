"""Upload a training video to Viam Data (viam-sdk 0.85).

Credentials come from API_KEY / API_KEY_ID, the names Viam training jobs set. Without them, or
without a part id, upload is skipped with a warning so local runs never fail on it.
"""

from __future__ import annotations

import asyncio
import os
import warnings

from viam.app.viam_client import ViamClient
from viam.rpc.dial import DialOptions


async def upload_video_async(
    path: str,
    part_id: str | None,
    tags: list[str] | None = None,
    dataset_ids: list[str] | None = None,
    timeout: float = 300,
) -> str | None:
    key, key_id = os.environ.get("API_KEY"), os.environ.get("API_KEY_ID")
    if not (key and key_id and part_id):
        warnings.warn(f"Viam upload skipped for {path}: API_KEY, API_KEY_ID and part_id are all required")
        return None

    async def go() -> str | None:
        client = await ViamClient.create_from_dial_options(DialOptions.with_api_key(key, key_id))
        try:
            return await client.data_client.file_upload_from_path(
                path, part_id=part_id, tags=tags, dataset_ids=dataset_ids, mime_type="video/mp4"
            )  # the uploaded file's id
        finally:
            client.close()

    return await asyncio.wait_for(go(), timeout)  # a stalled upload raises TimeoutError


def upload_video(
    path: str,
    part_id: str | None,
    tags: list[str] | None = None,
    dataset_ids: list[str] | None = None,
    timeout: float = 300,
) -> str | None:
    """Blocking wrapper for scripts and the training loop. Returns the file id, or None if skipped.

    Raises on upload errors and TimeoutError after `timeout` s; callers that must not fail (training) catch and log.
    """
    return asyncio.run(upload_video_async(path, part_id, tags, dataset_ids, timeout))
