"""Tests for db/models.py — SQLAlchemy async ORM on SQLite."""

import uuid

import pytest
import pytest_asyncio

from db.models import (
    Detection,
    Identification,
    Observation,
    get_engine,
    get_session_factory,
    init_db,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine():
    """In-memory SQLite engine scoped to one test."""
    eng = get_engine("sqlite+aiosqlite:///:memory:")
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    """AsyncSession bound to the in-memory engine."""
    factory = get_session_factory(engine)
    async with factory() as sess:
        yield sess


def _obs_id() -> str:
    return str(uuid.uuid4())


def _det_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schema_creates_without_error(engine):
    """init_db() should create all tables on a fresh SQLite engine."""
    # If we reach here the fixture already ran init_db — just assert no error.
    from sqlalchemy import inspect

    async with engine.connect() as conn:
        table_names = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )

    expected = {
        "observations",
        "detections",
        "identifications",
        "tracklets",
        "tracklet_detections",
    }
    assert expected.issubset(set(table_names))


@pytest.mark.asyncio
async def test_lookup_indexes_created(engine):
    """init_db() should create indexes used by result polling."""
    from sqlalchemy import inspect

    async with engine.connect() as conn:
        indexes = await conn.run_sync(
            lambda sync_conn: {
                (table, index["name"])
                for table in ("detections", "identifications")
                for index in inspect(sync_conn).get_indexes(table)
            }
        )

    assert ("detections", "idx_detections_observation_id") in indexes
    assert ("identifications", "idx_identifications_detection_id") in indexes


@pytest.mark.asyncio
async def test_observation_insert_and_query_by_id(session):
    """Inserting an Observation and fetching it by primary key returns the same row."""
    obs_id = _obs_id()
    obs = Observation(
        id=obs_id,
        filename="test_image.fits",
        obs_epoch="2024-01-15T03:22:00",
        exposure_time=30.0,
        status="queued",
    )
    session.add(obs)
    await session.commit()

    result = await session.get(Observation, obs_id)
    assert result is not None
    assert result.filename == "test_image.fits"
    assert result.exposure_time == 30.0
    assert result.obs_epoch == "2024-01-15T03:22:00"
    assert result.status == "queued"


@pytest.mark.asyncio
async def test_detection_references_observation(session):
    """A Detection row must reference its parent Observation by foreign key."""
    obs_id = _obs_id()
    session.add(Observation(id=obs_id, filename="img.fits", status="complete"))
    await session.flush()

    det_id = _det_id()
    det = Detection(
        id=det_id,
        observation_id=obs_id,
        confidence=0.93,
        x1=100.0, y1=200.0,
        x2=400.0, y2=220.0,
        angle_deg=3.814,
        streak_length_px=300.666,
        ra_tip1_deg=123.45, dec_tip1_deg=-10.2,
        ra_tip2_deg=123.50, dec_tip2_deg=-10.3,
    )
    session.add(det)
    await session.commit()

    fetched = await session.get(Detection, det_id)
    assert fetched is not None
    assert fetched.observation_id == obs_id
    assert fetched.method == "ml"
    assert fetched.confidence == pytest.approx(0.93)
    assert fetched.ra_tip1_deg == pytest.approx(123.45)


@pytest.mark.asyncio
async def test_identification_references_detection(session):
    """An Identification row must be linked to a Detection."""
    obs_id = _obs_id()
    session.add(Observation(id=obs_id, filename="img.fits", status="complete"))
    await session.flush()

    det_id = _det_id()
    session.add(Detection(id=det_id, observation_id=obs_id, confidence=0.88))
    await session.flush()

    ident_id = str(uuid.uuid4())
    ident = Identification(
        id=ident_id,
        detection_id=det_id,
        norad_id=25544,
        satellite_name="ISS (ZARYA)",
        confidence=0.91,
        separation_deg=0.003,
        rank=1,
        atrk_arcsec=12.5,
        xtrk_arcsec=-3.25,
        rotation_score=0.98,
        lateral_score=0.999,
        epoch_penalty=0.91,
        confidence_method="rotation_x_lateral_x_tle_age",
    )
    session.add(ident)
    await session.commit()

    fetched = await session.get(Identification, ident_id)
    assert fetched is not None
    assert fetched.detection_id == det_id
    assert fetched.norad_id == 25544
    assert fetched.rank == 1
    assert fetched.confidence == pytest.approx(0.91)
    assert fetched.atrk_arcsec == pytest.approx(12.5)
    assert fetched.xtrk_arcsec == pytest.approx(-3.25)
    assert fetched.rotation_score == pytest.approx(0.98)
    assert fetched.lateral_score == pytest.approx(0.999)
    assert fetched.epoch_penalty == pytest.approx(0.91)
    assert fetched.confidence_method == "rotation_x_lateral_x_tle_age"


@pytest.mark.asyncio
async def test_status_transitions(session):
    """Observation status should transition queued → processing → complete."""
    obs_id = _obs_id()
    session.add(Observation(id=obs_id, filename="img.fits", status="queued"))
    await session.commit()

    # queued → processing
    obs = await session.get(Observation, obs_id)
    obs.status = "processing"
    await session.commit()

    obs = await session.get(Observation, obs_id)
    assert obs.status == "processing"

    # processing → complete
    obs.status = "complete"
    await session.commit()

    obs = await session.get(Observation, obs_id)
    assert obs.status == "complete"
