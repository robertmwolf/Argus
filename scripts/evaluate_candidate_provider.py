"""Compare ARGUS cross-ID candidate providers on the same detections.

This script benchmarks candidate-source behavior rather than detector quality.
For each FITS image it:

1. Generates detections once with the ARGUS inference pipeline.
2. Loads the observation metadata from the FITS header.
3. Fetches candidate catalogs from each configured provider.
4. Scores the same detections against each provider's candidates.
5. Writes JSON and Markdown summaries.

Examples:

    python scripts/evaluate_candidate_provider.py \
        --fits data/sample/synth_streak_000.fits \
        --providers local satchecker \
        --out results/candidate_provider_eval.json

    python scripts/evaluate_candidate_provider.py \
        --manifest data/eval_fits.txt \
        --detection-mode full \
        --max-detections 3
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDERS = ["local", "satchecker"]


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    """Temporarily set or unset environment variables."""
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _read_manifest(path: Path) -> list[Path]:
    """Read a text or JSON manifest of FITS paths."""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return [Path(str(item)) for item in payload]
        if isinstance(payload, dict):
            for key in ("fits", "files", "images"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [Path(str(item)) for item in value]
        raise ValueError(f"JSON manifest {path} must be a list or contain fits/files/images")

    paths: list[Path] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        paths.append(Path(stripped))
    return paths


def _resolve_fits_paths(args: argparse.Namespace) -> list[Path]:
    """Return all FITS paths requested by the CLI."""
    paths: list[Path] = []
    for item in args.fits or []:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.fit*")))
        else:
            paths.append(p)
    if args.manifest:
        paths.extend(_read_manifest(Path(args.manifest)))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _load_fits_metadata(path: Path) -> dict[str, Any]:
    """Load observation metadata needed by cross-ID."""
    from inference.fits_loader import FITSLoader

    loaded = FITSLoader().load(path)
    obs_time = loaded.get("obs_time")
    if obs_time is None:
        obs_time = datetime.now(tz=timezone.utc)
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)
    return {
        "obs_time": obs_time,
        "observer_lat": loaded.get("observer_lat") or 0.0,
        "observer_lon": loaded.get("observer_lon") or 0.0,
        "observer_alt_m": loaded.get("observer_alt_m") or 0.0,
        "exposure_time": loaded.get("exposure_time"),
        "wcs_source": loaded.get("wcs_source"),
        "shape": loaded.get("shape"),
    }


def _generate_detections(path: Path, detection_mode: str, raw_mode: bool) -> list[dict[str, Any]]:
    """Run ARGUS once to generate detections without doing provider cross-ID."""
    from inference.pipeline import run

    if detection_mode == "fast":
        detections = run(path, fast=True, raw_mode=raw_mode)
    elif detection_mode == "full":
        with _temporary_env({"CROSSID_MAX_DETECTIONS": "0"}):
            detections = run(path, fast=False, raw_mode=raw_mode)
    else:
        raise ValueError(f"Unknown detection mode: {detection_mode}")

    for det in detections:
        det["identifications"] = []
    return detections


def _load_detection_fixture(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load precomputed detections plus optional metadata from JSON."""
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict):
        raise ValueError(f"Detection fixture {path} must be a JSON list or object")
    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise ValueError(f"Detection fixture {path} missing list field 'detections'")
    meta = payload.get("metadata") or {}
    if "obs_time" in meta and isinstance(meta["obs_time"], str):
        meta["obs_time"] = datetime.fromisoformat(meta["obs_time"].replace("Z", "+00:00"))
    return detections, meta


def _candidate_key(ident: dict[str, Any]) -> int | None:
    """Return a normalized NORAD id from an identification row."""
    value = ident.get("norad_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _top_ids(detections: list[dict[str, Any]], limit: int = 3) -> list[int]:
    """Return unique top NORAD ids across detections, preserving score order."""
    ranked: list[dict[str, Any]] = []
    for det in detections:
        ranked.extend(det.get("identifications", []))
    ranked.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    ids: list[int] = []
    seen: set[int] = set()
    for ident in ranked:
        norad_id = _candidate_key(ident)
        if norad_id is None or norad_id in seen:
            continue
        seen.add(norad_id)
        ids.append(norad_id)
        if len(ids) >= limit:
            break
    return ids


def _best_identification(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-confidence identification across all detections."""
    best: dict[str, Any] | None = None
    best_score = -1.0
    for det in detections:
        for ident in det.get("identifications", []):
            score = float(ident.get("confidence") or 0.0)
            if score > best_score:
                best_score = score
                best = ident
    return best


def _catalog_sources(catalog: list[dict[str, Any]]) -> list[str]:
    """Return sorted provider/source labels from candidate catalog entries."""
    sources = {
        str(entry.get("tle_search_mode") or entry.get("source") or "unknown")
        for entry in catalog
    }
    return sorted(sources)


def evaluate_provider(
    provider: str,
    detections: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    epoch_window_days: int,
    min_mean_motion: float,
    max_detections: int,
) -> dict[str, Any]:
    """Fetch candidates and score detections for one provider."""
    from inference.crossid import _fetch_candidate_catalog, cross_identify

    scoped_detections = sorted(
        copy.deepcopy(detections),
        key=lambda det: float(det.get("confidence") or 0.0),
        reverse=True,
    )[:max_detections]

    obs_time = metadata["obs_time"]
    t0 = time.perf_counter()
    fetch_error: str | None = None
    catalog: list[dict[str, Any]] = []
    with _temporary_env({"ARGUS_CANDIDATE_PROVIDER": provider}):
        try:
            catalog = _fetch_candidate_catalog(
                scoped_detections,
                obs_time,
                epoch_window_days,
                min_mean_motion,
                float(metadata.get("observer_lat") or 0.0),
                float(metadata.get("observer_lon") or 0.0),
                float(metadata.get("observer_alt_m") or 0.0),
                metadata.get("exposure_time"),
            )
        except Exception as exc:
            fetch_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Candidate fetch failed provider=%s: %s", provider, fetch_error)
    fetch_ms = (time.perf_counter() - t0) * 1000.0

    score_ms = 0.0
    if catalog:
        t1 = time.perf_counter()
        cross_identify(
            scoped_detections,
            obs_time,
            float(metadata.get("observer_lat") or 0.0),
            float(metadata.get("observer_lon") or 0.0),
            float(metadata.get("observer_alt_m") or 0.0),
            epoch_window_days=epoch_window_days,
            exposure_time=metadata.get("exposure_time"),
            min_mean_motion=min_mean_motion,
            _catalog_override=catalog,
            _allow_current_retry=False,
        )
        score_ms = (time.perf_counter() - t1) * 1000.0
    else:
        for det in scoped_detections:
            det["identifications"] = []

    best = _best_identification(scoped_detections)
    top3 = _top_ids(scoped_detections, limit=3)
    return {
        "provider": provider,
        "candidate_count": len(catalog),
        "catalog_sources": _catalog_sources(catalog),
        "fetch_ms": round(fetch_ms, 1),
        "score_ms": round(score_ms, 1),
        "total_ms": round(fetch_ms + score_ms, 1),
        "error": fetch_error,
        "top1_norad": _candidate_key(best) if best else None,
        "top1_name": best.get("satellite_name") if best else None,
        "top1_confidence": round(float(best.get("confidence") or 0.0), 6) if best else None,
        "top3_norads": top3,
        "identification_count": sum(len(det.get("identifications", [])) for det in scoped_detections),
        "detections": scoped_detections,
    }


def _compare_to_baseline(provider_rows: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    """Compute per-image comparison metrics against the baseline provider."""
    by_provider = {row["provider"]: row for row in provider_rows}
    base = by_provider.get(baseline)
    if base is None:
        return {}
    base_top1 = base.get("top1_norad")
    base_top3 = set(base.get("top3_norads") or [])
    comparisons: dict[str, Any] = {}
    for provider, row in by_provider.items():
        if provider == baseline:
            continue
        top3 = set(row.get("top3_norads") or [])
        confidence_delta = None
        if row.get("top1_confidence") is not None and base.get("top1_confidence") is not None:
            confidence_delta = round(row["top1_confidence"] - base["top1_confidence"], 6)
        comparisons[provider] = {
            "top1_match_same": row.get("top1_norad") == base_top1 and base_top1 is not None,
            "top3_contains_baseline_top1": base_top1 in top3 if base_top1 is not None else False,
            "top3_overlap": sorted(base_top3 & top3),
            "confidence_delta": confidence_delta,
            "empty_catalog": row.get("candidate_count", 0) == 0,
            "error": row.get("error"),
        }
    return comparisons


def evaluate_image(
    path: Path,
    providers: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate all providers for one FITS image or detection fixture."""
    if args.detections_json:
        detections, fixture_meta = _load_detection_fixture(Path(args.detections_json))
        metadata = {**_load_fits_metadata(path), **fixture_meta}
    else:
        metadata = _load_fits_metadata(path)
        detections = _generate_detections(path, args.detection_mode, args.raw_mode)

    if not detections:
        logger.warning("No detections for %s", path)

    rows = [
        evaluate_provider(
            provider,
            detections,
            metadata,
            epoch_window_days=args.epoch_window_days,
            min_mean_motion=args.min_mean_motion,
            max_detections=args.max_detections,
        )
        for provider in providers
    ]

    return {
        "filename": str(path),
        "obs_time": metadata["obs_time"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "observer_lat": metadata.get("observer_lat"),
        "observer_lon": metadata.get("observer_lon"),
        "observer_alt_m": metadata.get("observer_alt_m"),
        "exposure_time": metadata.get("exposure_time"),
        "wcs_source": metadata.get("wcs_source"),
        "detection_count": len(detections),
        "provider_results": rows,
        "comparisons": _compare_to_baseline(rows, args.baseline_provider),
    }


def _mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 3) if vals else None


def summarize(results: list[dict[str, Any]], providers: list[str], baseline: str) -> dict[str, Any]:
    """Aggregate provider comparison metrics across all images."""
    provider_summary: dict[str, Any] = {}
    for provider in providers:
        rows = [
            row
            for image in results
            for row in image["provider_results"]
            if row["provider"] == provider
        ]
        provider_summary[provider] = {
            "images": len(rows),
            "empty_catalog_rate": _mean([1.0 if row["candidate_count"] == 0 else 0.0 for row in rows]),
            "error_rate": _mean([1.0 if row["error"] else 0.0 for row in rows]),
            "mean_candidate_count": _mean([float(row["candidate_count"]) for row in rows]),
            "mean_fetch_ms": _mean([float(row["fetch_ms"]) for row in rows]),
            "mean_total_ms": _mean([float(row["total_ms"]) for row in rows]),
            "identified_rate": _mean([1.0 if row["top1_norad"] is not None else 0.0 for row in rows]),
        }

    comparison_summary: dict[str, Any] = {}
    for provider in providers:
        if provider == baseline:
            continue
        comparisons = [
            image["comparisons"].get(provider)
            for image in results
            if image.get("comparisons", {}).get(provider) is not None
        ]
        comparison_summary[provider] = {
            "images": len(comparisons),
            "top1_agreement_rate": _mean([
                1.0 if item["top1_match_same"] else 0.0 for item in comparisons
            ]),
            "top3_contains_baseline_top1_rate": _mean([
                1.0 if item["top3_contains_baseline_top1"] else 0.0
                for item in comparisons
            ]),
            "mean_confidence_delta": _mean([
                item["confidence_delta"]
                for item in comparisons
                if item["confidence_delta"] is not None
            ]),
        }

    return {
        "baseline_provider": baseline,
        "provider_summary": provider_summary,
        "comparison_summary": comparison_summary,
    }


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact Markdown summary next to the JSON output."""
    lines = [
        "# Candidate Provider Evaluation",
        "",
        f"Baseline provider: `{payload['summary']['baseline_provider']}`",
        f"Images: {len(payload['images'])}",
        "",
        "## Provider Summary",
        "",
        "| Provider | Empty Rate | Error Rate | Mean Candidates | Mean Fetch ms | Identified Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for provider, stats in payload["summary"]["provider_summary"].items():
        lines.append(
            f"| `{provider}` | {stats['empty_catalog_rate']} | {stats['error_rate']} | "
            f"{stats['mean_candidate_count']} | {stats['mean_fetch_ms']} | "
            f"{stats['identified_rate']} |"
        )

    lines.extend([
        "",
        "## Baseline Comparison",
        "",
        "| Provider | Top-1 Agreement | Top-3 Contains Baseline Top-1 | Mean Confidence Delta |",
        "|---|---:|---:|---:|",
    ])
    for provider, stats in payload["summary"]["comparison_summary"].items():
        lines.append(
            f"| `{provider}` | {stats['top1_agreement_rate']} | "
            f"{stats['top3_contains_baseline_top1_rate']} | "
            f"{stats['mean_confidence_delta']} |"
        )

    lines.extend(["", "## Per Image", ""])
    for image in payload["images"]:
        lines.append(f"### `{image['filename']}`")
        lines.append("")
        lines.append(
            f"- obs_time: `{image['obs_time']}`; detections: {image['detection_count']}; "
            f"wcs_source: `{image.get('wcs_source')}`"
        )
        for row in image["provider_results"]:
            lines.append(
                f"- `{row['provider']}`: candidates={row['candidate_count']}, "
                f"top1={row['top1_norad']} {row['top1_name'] or ''}, "
                f"conf={row['top1_confidence']}, total_ms={row['total_ms']}, "
                f"error={row['error']}"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local vs SatChecker ARGUS cross-ID candidate providers."
    )
    parser.add_argument("--fits", nargs="*", help="FITS file(s) or directories to evaluate")
    parser.add_argument("--manifest", help="Text or JSON manifest of FITS paths")
    parser.add_argument("--detections-json", help="Optional precomputed detection fixture JSON")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=_DEFAULT_PROVIDERS,
        help="Candidate providers to compare",
    )
    parser.add_argument("--baseline-provider", default="local")
    parser.add_argument("--out", default="results/candidate_provider_eval.json")
    parser.add_argument("--detection-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--raw-mode", action="store_true")
    parser.add_argument("--max-detections", type=int, default=3)
    parser.add_argument("--epoch-window-days", type=int, default=3)
    parser.add_argument("--min-mean-motion", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    fits_paths = _resolve_fits_paths(args)
    if not fits_paths:
        raise SystemExit("Provide at least one --fits path or --manifest")
    if args.baseline_provider not in args.providers:
        raise SystemExit("--baseline-provider must be included in --providers")

    images = [evaluate_image(path, args.providers, args) for path in fits_paths]
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "providers": args.providers,
            "baseline_provider": args.baseline_provider,
            "detection_mode": args.detection_mode,
            "raw_mode": args.raw_mode,
            "max_detections": args.max_detections,
            "epoch_window_days": args.epoch_window_days,
            "min_mean_motion": args.min_mean_motion,
        },
        "summary": summarize(images, args.providers, args.baseline_provider),
        "images": images,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    md_path = out_path.with_suffix(".md")
    write_markdown_report(md_path, payload)

    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
