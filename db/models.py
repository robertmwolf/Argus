"""SQLAlchemy 2.0 async ORM models for ARGUS.

Supports PostgreSQL 16 (asyncpg driver) and SQLite (aiosqlite driver).
Driver is selected via the DATABASE_URL environment variable:
  sqlite+aiosqlite:///./argus.db   (default, local dev)
  postgresql+asyncpg://user:pass@host/db  (production)
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy import Float, ForeignKey, Integer, Text, inspect
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./argus.db"


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)


def get_engine(database_url: str | None = None) -> AsyncEngine:
    """Create an async engine from DATABASE_URL (or override).

    Args:
        database_url: Override the DATABASE_URL env var.

    Returns:
        Configured AsyncEngine instance.
    """
    url = database_url or get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_async_engine(url, connect_args=connect_args)


def get_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session factory bound to the given engine.

    Args:
        engine: AsyncEngine to bind sessions to.

    Returns:
        async_sessionmaker that produces AsyncSession instances.
    """
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_async_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Async generator yielding a single session — for use as a FastAPI dependency.

    Args:
        session_factory: Factory produced by get_session_factory().

    Yields:
        AsyncSession scoped to one request.
    """
    async with session_factory() as session:
        yield session


class Base(DeclarativeBase):
    pass


class Observation(Base):
    """One uploaded FITS file and its processing state."""

    __tablename__ = "observations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    uploaded_at: Mapped[str | None] = mapped_column(Text)
    exposure_time: Mapped[float | None] = mapped_column(Float)
    obs_epoch: Mapped[str | None] = mapped_column(Text)       # ISO8601
    fits_wcs_json: Mapped[str | None] = mapped_column(Text)   # JSON string
    status: Mapped[str] = mapped_column(Text, default="queued", nullable=False)


class Detection(Base):
    """One streak detection within an observation."""

    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    observation_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("observations.id")
    )
    method: Mapped[str] = mapped_column(Text, default="ml", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x1: Mapped[float | None] = mapped_column(Float)
    bbox_y1: Mapped[float | None] = mapped_column(Float)
    bbox_x2: Mapped[float | None] = mapped_column(Float)
    bbox_y2: Mapped[float | None] = mapped_column(Float)
    obb_cx: Mapped[float | None] = mapped_column(Float)
    obb_cy: Mapped[float | None] = mapped_column(Float)
    obb_w: Mapped[float | None] = mapped_column(Float)
    obb_h: Mapped[float | None] = mapped_column(Float)
    obb_angle_deg: Mapped[float | None] = mapped_column(Float)
    streak_length_px: Mapped[float | None] = mapped_column(Float)
    streak_id: Mapped[int | None] = mapped_column(Integer)
    ra_tip1_deg: Mapped[float | None] = mapped_column(Float)
    dec_tip1_deg: Mapped[float | None] = mapped_column(Float)
    ra_tip2_deg: Mapped[float | None] = mapped_column(Float)
    dec_tip2_deg: Mapped[float | None] = mapped_column(Float)


class Identification(Base):
    """Satellite candidate matched to a detection (up to 3 per detection)."""

    __tablename__ = "identifications"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    detection_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("detections.id")
    )
    norad_id: Mapped[int | None] = mapped_column(Integer)
    satellite_name: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    separation_deg: Mapped[float | None] = mapped_column(Float)
    rank: Mapped[int | None] = mapped_column(Integer)  # 1 = best


class Tracklet(Base):
    """Group of detections linked across frames."""

    __tablename__ = "tracklets"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[str | None] = mapped_column(Text)


class TrackletDetection(Base):
    """Association between a tracklet and a detection."""

    __tablename__ = "tracklet_detections"

    tracklet_id: Mapped[str] = mapped_column(
        Text, ForeignKey("tracklets.id"), primary_key=True
    )
    detection_id: Mapped[str] = mapped_column(
        Text, ForeignKey("detections.id"), primary_key=True
    )
    frame_index: Mapped[int | None] = mapped_column(Integer)


class TleCatalogEntry(Base):
    """One TLE epoch for one catalogued object.

    Populated at environment setup from Space-Track annual zip bundles and
    kept current by the hourly GP-class updater.  Never re-fetched from
    gp_history once stored here.
    """

    __tablename__ = "tle_catalog"

    norad_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    epoch: Mapped[str] = mapped_column(Text, primary_key=True)  # ISO8601 UTC
    object_name: Mapped[str] = mapped_column(Text, nullable=False)
    object_type: Mapped[str | None] = mapped_column(Text)
    mean_motion: Mapped[float | None] = mapped_column(Float)    # rev/day
    tle_line1: Mapped[str] = mapped_column(Text, nullable=False)
    tle_line2: Mapped[str] = mapped_column(Text, nullable=False)
    ingested_at: Mapped[str | None] = mapped_column(Text)


class TleCatalogCoverage(Base):
    """Records which data sources have been loaded into tle_catalog.

    Allows bootstrap_tle_catalog.py to skip years already present so
    re-running the script is always a safe no-op.
    """

    __tablename__ = "tle_catalog_coverage"

    source_tag: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[str | None] = mapped_column(Text)
    record_count: Mapped[int | None] = mapped_column(Integer)
    downloaded_at: Mapped[str | None] = mapped_column(Text)


async def init_db(engine: AsyncEngine) -> None:
    """Create all tables if they do not already exist.

    Args:
        engine: AsyncEngine to create tables on.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_existing_tables)


def _migrate_existing_tables(sync_conn) -> None:
    """Apply lightweight schema additions for existing local databases."""
    columns = {col["name"] for col in inspect(sync_conn).get_columns("detections")}
    if "method" not in columns:
        sync_conn.exec_driver_sql("ALTER TABLE detections ADD COLUMN method TEXT DEFAULT 'ml'")
    if "streak_id" not in columns:
        sync_conn.exec_driver_sql("ALTER TABLE detections ADD COLUMN streak_id INTEGER")


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    async def _smoke_test() -> None:
        engine = get_engine("sqlite+aiosqlite:///./argus_smoke.db")
        await init_db(engine)
        logger.info("Tables created successfully.")
        await engine.dispose()

        import os
        os.unlink("argus_smoke.db")
        logger.info("Smoke test passed.")

    asyncio.run(_smoke_test())
