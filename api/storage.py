"""Storage backend abstraction for ARGUS uploads and processed images.

Selected via STORAGE_BACKEND env var:
  local  — saves files under data/uploads/ (default)
  s3     — AWS S3 (stub, implemented in Phase 7)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Abstract storage backend for FITS uploads and rendered PNGs."""

    @abstractmethod
    async def save_upload(self, job_id: str, filename: str, data: bytes) -> Path:
        """Persist raw upload bytes and return the file path.

        Args:
            job_id: UUID string identifying the job.
            filename: Original filename from the upload.
            data: Raw file bytes.

        Returns:
            Path where the file was written.
        """

    @abstractmethod
    async def load_upload(self, job_id: str, filename: str) -> bytes:
        """Load previously saved upload bytes.

        Args:
            job_id: UUID string identifying the job.
            filename: Original filename used in save_upload.

        Returns:
            Raw file bytes.
        """

    @abstractmethod
    async def save_image(self, job_id: str, data: bytes) -> None:
        """Persist the processed result PNG.

        Args:
            job_id: UUID string identifying the job.
            data: PNG bytes to store.
        """

    @abstractmethod
    async def load_image(self, job_id: str) -> bytes | None:
        """Load the processed result PNG, or None if not yet available.

        Args:
            job_id: UUID string identifying the job.

        Returns:
            PNG bytes, or None if not found.
        """

    @abstractmethod
    async def save_preview(self, job_id: str, data: bytes) -> None:
        """Persist a raw preview PNG (no detection overlays).

        Args:
            job_id: UUID string identifying the job.
            data: PNG bytes to store.
        """

    @abstractmethod
    async def load_preview(self, job_id: str) -> bytes | None:
        """Load the raw preview PNG, or None if not yet available.

        Args:
            job_id: UUID string identifying the job.

        Returns:
            PNG bytes, or None if not found.
        """

    @abstractmethod
    async def save_fits_header(self, job_id: str, data: bytes) -> None:
        """Persist serialised FITS header JSON.

        Args:
            job_id: UUID string identifying the job.
            data: UTF-8 encoded JSON bytes.
        """

    @abstractmethod
    async def load_fits_header(self, job_id: str) -> bytes | None:
        """Load serialised FITS header JSON, or None if not available.

        Args:
            job_id: UUID string identifying the job.

        Returns:
            UTF-8 encoded JSON bytes, or None if not found.
        """


class LocalStorage(StorageBackend):
    """Stores files under a local base directory."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self._base = Path(base_dir) if base_dir else Path("data/uploads")

    async def save_upload(self, job_id: str, filename: str, data: bytes) -> Path:
        dest = self._base / job_id / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    async def load_upload(self, job_id: str, filename: str) -> bytes:
        return (self._base / job_id / filename).read_bytes()

    async def save_image(self, job_id: str, data: bytes) -> None:
        path = self._base / job_id / "result.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def load_image(self, job_id: str) -> bytes | None:
        path = self._base / job_id / "result.png"
        return path.read_bytes() if path.exists() else None

    async def save_preview(self, job_id: str, data: bytes) -> None:
        path = self._base / job_id / "preview.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def load_preview(self, job_id: str) -> bytes | None:
        path = self._base / job_id / "preview.png"
        return path.read_bytes() if path.exists() else None

    async def save_fits_header(self, job_id: str, data: bytes) -> None:
        path = self._base / job_id / "header.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def load_fits_header(self, job_id: str) -> bytes | None:
        path = self._base / job_id / "header.json"
        return path.read_bytes() if path.exists() else None


class S3Storage(StorageBackend):
    """AWS S3 storage backend — implemented in Phase 7."""

    async def save_upload(self, job_id: str, filename: str, data: bytes) -> Path:
        raise NotImplementedError("S3 storage not implemented until Phase 7")

    async def load_upload(self, job_id: str, filename: str) -> bytes:
        raise NotImplementedError("S3 storage not implemented until Phase 7")

    async def save_image(self, job_id: str, data: bytes) -> None:
        raise NotImplementedError("S3 storage not implemented until Phase 7")

    async def load_image(self, job_id: str) -> bytes | None:
        raise NotImplementedError("S3 storage not implemented until Phase 7")

    async def save_preview(self, job_id: str, data: bytes) -> None:
        raise NotImplementedError("S3 storage not implemented until Phase 7")

    async def load_preview(self, job_id: str) -> bytes | None:
        raise NotImplementedError("S3 storage not implemented until Phase 7")

    async def save_fits_header(self, job_id: str, data: bytes) -> None:
        raise NotImplementedError("S3 storage not implemented until Phase 7")

    async def load_fits_header(self, job_id: str) -> bytes | None:
        raise NotImplementedError("S3 storage not implemented until Phase 7")


def get_storage(base_dir: Path | str | None = None) -> StorageBackend:
    """Factory: return storage backend selected by STORAGE_BACKEND env var.

    Args:
        base_dir: Override base directory for LocalStorage (tests only).

    Returns:
        Configured StorageBackend instance.

    Raises:
        ValueError: If STORAGE_BACKEND is not a known value.
    """
    backend = os.environ.get("STORAGE_BACKEND", "local").lower()
    if backend == "local":
        return LocalStorage(base_dir)
    if backend == "s3":
        return S3Storage()
    raise ValueError(f"Unknown STORAGE_BACKEND: {backend!r}")


if __name__ == "__main__":
    import asyncio
    import tempfile

    async def _smoke() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStorage(tmp)
            await store.save_upload("job-1", "test.fits", b"FITS_BYTES")
            data = await store.load_upload("job-1", "test.fits")
            assert data == b"FITS_BYTES"
            await store.save_image("job-1", b"PNG_BYTES")
            img = await store.load_image("job-1")
            assert img == b"PNG_BYTES"
            print("LocalStorage smoke test passed.")

    asyncio.run(_smoke())
