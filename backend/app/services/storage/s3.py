import re
from pathlib import Path

import aioboto3
from aiobotocore.config import AioConfig
from botocore.exceptions import ClientError


_DOSPACES_BUCKET_SUBDOMAIN_RE = re.compile(
    r"^(https?://)([a-z0-9.\-]+)\.([a-z0-9-]+\.digitaloceanspaces\.com)/?$"
)


def _normalize_endpoint(endpoint_url: str, bucket_name: str) -> str:
    """Return a regional DO Spaces endpoint, stripping the bucket subdomain
    if present.

    DO Spaces has two URLs that look almost identical: the bucket "origin
    endpoint" (bucket.region.digitaloceanspaces.com) and the regional API
    endpoint (region.digitaloceanspaces.com). boto3 expects the regional
    one — handing it the bucket-included URL makes boto3 double-prefix
    the bucket into the key path, leaving every object at
    `<bucket>/audio/...` instead of `audio/...`. Be tolerant: accept
    either form."""
    m = _DOSPACES_BUCKET_SUBDOMAIN_RE.match(endpoint_url)
    if m and m.group(2) == bucket_name:
        return f"{m.group(1)}{m.group(3)}"
    return endpoint_url

# Presigned-URL lifetime. 1 h is comfortably longer than the longest expected
# audio play, short enough that a leaked link from the dev console expires
# before the listener could share it widely.
PRESIGNED_URL_TTL_SECONDS = 3600


class S3Storage:
    """S3-compatible StorageBackend using aioboto3.

    Keys are mapped directly to objects in the bucket. `get_url` returns a
    short-lived presigned HTTPS URL suitable for handing back to a browser
    via RedirectResponse.
    """

    def __init__(
        self,
        endpoint_url: str,
        bucket_name: str,
        access_key: str,
        secret_key: str,
        region_name: str = "auto",
    ) -> None:
        self.endpoint_url = _normalize_endpoint(endpoint_url, bucket_name)
        self.bucket_name = bucket_name
        self.access_key = access_key
        self.secret_key = secret_key
        self.region_name = region_name
        self.session = aioboto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
        )
        # Virtual-hosted addressing + sigv4 produce clean URLs of the
        # form `https://<bucket>.<region>.digitaloceanspaces.com/<key>`.
        # Without this boto3 falls back to path-style on custom endpoints
        # and the bucket name leaks into the object key path.
        self._config = AioConfig(
            s3={"addressing_style": "virtual"},
            signature_version="s3v4",
        )

    def _client(self):
        return self.session.client(
            "s3", endpoint_url=self.endpoint_url, config=self._config,
        )

    async def write(self, key: str, data: bytes) -> str:
        async with self._client() as client:
            await client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
            )
        return key

    async def read(self, key: str) -> bytes:
        async with self._client() as client:
            try:
                response = await client.get_object(Bucket=self.bucket_name, Key=key)
                async with response["Body"] as stream:
                    return await stream.read()
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    raise FileNotFoundError(f"Key not found: {key}")
                raise

    async def exists(self, key: str) -> bool:
        async with self._client() as client:
            try:
                await client.head_object(Bucket=self.bucket_name, Key=key)
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False
                raise

    async def delete(self, key: str) -> None:
        async with self._client() as client:
            await client.delete_object(Bucket=self.bucket_name, Key=key)

    async def get_url(self, key: str) -> str:
        """Return a short-lived presigned HTTPS URL for `key`.

        The API uses this to RedirectResponse the browser straight to
        DO Spaces — no audio traffic crosses the backend.
        """
        async with self._client() as client:
            return await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=PRESIGNED_URL_TTL_SECONDS,
            )

    async def download_file(self, key: str, dest_path: Path) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._client() as client:
            try:
                await client.download_file(self.bucket_name, key, str(dest_path))
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    raise FileNotFoundError(f"Key not found: {key}")
                raise

    async def upload_file(self, src_path: Path, key: str) -> str:
        async with self._client() as client:
            await client.upload_file(str(src_path), self.bucket_name, key)
        return key

    async def stream(self, key: str, *, start=None, end=None):
        """Stream `key` from Spaces, optionally with a Range. Returns
        (async-iterator of chunks, total_size, content_length).

        We open the S3 client inside the iterator so it stays alive for
        the whole stream — closing it via `async with` would cancel the
        body. The caller (StreamingResponse) controls iteration; when it
        finishes (or the client disconnects), the iterator's `finally`
        closes the client cleanly."""
        kwargs: dict = {"Bucket": self.bucket_name, "Key": key}
        if start is not None or end is not None:
            lo = 0 if start is None else start
            hi = "" if end is None else str(end)
            kwargs["Range"] = f"bytes={lo}-{hi}"

        client_cm = self._client()
        client = await client_cm.__aenter__()
        try:
            try:
                response = await client.get_object(**kwargs)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("NoSuchKey", "404"):
                    await client_cm.__aexit__(None, None, None)
                    raise FileNotFoundError(f"Key not found: {key}")
                await client_cm.__aexit__(None, None, None)
                raise

            # Parse Content-Range when present ("bytes 0-99/28770382"),
            # otherwise rely on ContentLength (full GET).
            content_length = int(response["ContentLength"])
            content_range = response.get("ContentRange")
            if content_range:
                total = int(content_range.rsplit("/", 1)[-1])
            else:
                total = content_length

            body_stream = response["Body"]

            async def gen():
                try:
                    async for chunk in body_stream.iter_chunks(64 * 1024):
                        yield chunk
                finally:
                    body_stream.close()
                    await client_cm.__aexit__(None, None, None)

            return gen(), total, content_length
        except Exception:
            # Make sure we never leak the client on the unhappy paths.
            try:
                await client_cm.__aexit__(None, None, None)
            except Exception:
                pass
            raise
