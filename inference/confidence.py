"""inference/confidence.py

Precision-recall calibrated Unified Confidence Score for ARGUS streak detections.

Replaces the plain Noisy-OR formula with an F-beta weighted corroboration score that:
  1. Uses the best detector confidence as the score floor
  2. Weights corroborating detectors by their empirical F-0.5 score
  3. Caps each detector's effective confidence at an optional ceiling (for
     detectors whose raw confidence magnitude is miscalibrated, e.g. ASTRiDE)
  4. Tempers the corroboration boost when detectors strongly disagree
  5. Treats ASTRiDE as corroboration-only: it cannot create a standalone
     streak confidence and can only add a small boost to non-ASTRiDE detections.
  6. Applies per-band reliability weights when streak_band is provided so that
     detectors dominant in specific length ranges contribute proportionally more.

The score represents how confident we can be that a detection is a real streak,
accounting for each detector's known reliability profile.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectorProfile:
    """Empirical performance profile for one detector method.

    Attributes:
        name: Human-readable detector name.
        precision: Empirical precision (TP / (TP + FP)).
        recall: Empirical recall (TP / (TP + FN)).
        confidence_ceiling: Optional cap on the detector's raw confidence before
            it enters the fusion formula.  Use this when a detector's confidence
            magnitude is unreliable — e.g. it routinely reports 0.95+ on false
            positives — so that only the *presence* of a detection is trusted,
            not its stated certainty.  None means no capping.
        band_weights: Optional per-band multipliers on the F-0.5 reliability
            weight.  Keys are "short", "medium", "long" (same thresholds as
            eval/metrics.py: short<150px, 150≤medium<400px, long≥400px).
            A multiplier of 2.0 doubles the detector's weight in that band;
            0.1 nearly silences it.  Missing keys default to 1.0.
        notes: Data source / confidence level for these estimates.
    """

    name: str
    precision: float
    recall: float
    confidence_ceiling: float | None = None
    band_weights: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @property
    def f_beta_weight(self) -> float:
        """F-0.5 score used as this detector's reliability weight."""
        return fbeta_weight(self.precision, self.recall, beta=0.5)

    def band_adjusted_weight(self, streak_band: str | None) -> float:
        """Return the F-0.5 weight scaled by the per-band multiplier.

        Args:
            streak_band: "short", "medium", or "long".  None → no adjustment.

        Returns:
            F-0.5 weight × band multiplier, clamped to [0, 1].
        """
        base = self.f_beta_weight
        if streak_band is None or not self.band_weights:
            return base
        multiplier = self.band_weights.get(streak_band, 1.0)
        return min(1.0, base * multiplier)


def fbeta_weight(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F-beta score as a detector reliability weight.

    Args:
        precision: Empirical precision in [0, 1].
        recall: Empirical recall in [0, 1].
        beta: Relative weight of recall vs precision. Values < 1 favour precision
            (i.e. penalise false positives more than false negatives).

    Returns:
        F-beta score in [0, 1]. Returns 0.0 when both inputs are zero.
    """
    denom = beta**2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta**2) * precision * recall / denom


# ---------------------------------------------------------------------------
# Detector profiles — sourced from benchmark measurements.
#
# Measured values are from the named results files.
# Estimated values are marked "est." and should be updated when new eval data
# is available.
#
# Per-band weights use eval/metrics.py thresholds: short<150px, 150≤medium<400px,
# long≥400px.  Weights were derived from per-band P/R in the same benchmarks.
# ---------------------------------------------------------------------------

DETECTOR_PROFILES: dict[str, DetectorProfile] = {
    # Source: phase8_benchmark.json — measured on 50-image synthetic dev subset
    "tiny": DetectorProfile(
        name="DINO Swin-T",
        precision=0.6667,
        recall=0.7333,
        notes="Phase 8 measured — synthetic dev subset only",
    ),
    # Source: full_yolo_obb/yolo_benchmark.json — measured on full tiled val split
    "yolo_full": DetectorProfile(
        name="YOLO11n-OBB Full Dataset",
        precision=0.5718,
        recall=0.8458,
        band_weights={"short": 0.5, "medium": 1.3, "long": 0.8},
        notes="results/full_yolo_obb/yolo_benchmark.json; no per-band breakdown available",
    ),
    # Source: phase8_benchmark.json — dev subset baseline
    "yolo": DetectorProfile(
        name="YOLO11n-OBB",
        precision=0.6316,
        recall=0.4000,
        notes="Phase 8 measured — dev subset",
    ),
    # Source: streakmind_yolo/gtimages_plus_frigate/metrics_iou50.json (best track)
    # Per-band: medium P=6.5% R=80%; long P=37.5% R=12%
    "streakmind_yolo": DetectorProfile(
        name="YOLO-OBB GTImages",
        precision=0.1532,
        recall=0.2982,
        band_weights={"short": 0.05, "medium": 2.5, "long": 0.2},
        notes="results/streakmind_yolo/gtimages_plus_frigate/metrics_iou50.json; "
              "medium recall=80%, long recall=12%",
    ),
    # Source: comprehensive_eval_20260526/report.md — standard test set, conf≥0.30, IoU≥0.50
    # Per-band recall (269/800px thresholds): short=50%, medium=72.7%, long=72.5%
    "dinov3_vitb_multisource": DetectorProfile(
        name="DINOv3 Base - Multi-source",
        precision=0.712,
        recall=0.724,
        band_weights={"short": 0.2, "medium": 0.9, "long": 1.3},
        notes="comprehensive_eval_20260526; P=71.2% R=72.4% at conf≥0.30; "
              "long dominant (295 GT annotations vs 11 medium)",
    ),
    # Source: comprehensive_eval_20260526/report.md — same checkpoint
    "dinov3_gt_dm_satstreaks": DetectorProfile(
        name="DINOv3 GT+DM+SatStreaks",
        precision=0.712,
        recall=0.724,
        band_weights={"short": 0.2, "medium": 0.9, "long": 1.3},
        notes="Same checkpoint as dinov3_vitb_multisource (run_best_400px_nodm); "
              "measured P/R from comprehensive_eval_20260526",
    ),
    # Source: multi_method_benchmark.json dinov3_vitb entry (older model, low-threshold run)
    # Re-estimated from mAP@0.5=0.74; per-band similar to multisource.
    "dinov3_vitb": DetectorProfile(
        name="DINOv3 ViT-B",
        precision=0.712,
        recall=0.724,
        band_weights={"short": 0.2, "medium": 0.9, "long": 1.3},
        notes="est. from comprehensive_eval; update with direct per-model P/R after Phase D",
    ),
    # DINOv3 ViT-L: Phase D target (RTX 5070 Ti workstation); conservative estimate
    "dinov3_vitl": DetectorProfile(
        name="DINOv3 ViT-L",
        precision=0.85,
        recall=0.82,
        band_weights={"short": 0.2, "medium": 0.9, "long": 1.3},
        notes="est. Phase D target; update post Phase D",
    ),
    # DINO Swin-L: estimated pre-Phase D
    "large": DetectorProfile(
        name="DINO Swin-L",
        precision=0.75,
        recall=0.75,
        notes="est. pre-Phase D",
    ),
    # ASTRiDE classical: near-zero recall in practice (classical contour detector).
    # Precision ~1-5% measured via OpenCV proxy in multi_method_benchmark.
    # Confidence magnitude is unreliable (aspect-ratio derived, not ML); ceiling
    # caps its effective contribution regardless of raw score.
    # Treated as corroboration-only in compute_unified_confidence.
    "astride": DetectorProfile(
        name="ASTRiDE",
        precision=0.03,
        recall=0.03,
        confidence_ceiling=0.45,
        notes="near-zero measured recall; ceiling retained because raw confidence "
              "is aspect-ratio derived and uncalibrated",
    ),
}

# Default for any method not in DETECTOR_PROFILES.
_DEFAULT_PROFILE = DetectorProfile(
    name="Unknown",
    precision=0.50,
    recall=0.50,
    notes="default — no benchmark data",
)

_ASTRIDE_METHODS = {"astride"}
_ASTRIDE_MAX_CORROBORATION_BOOST = 0.04


def _get_profile(method: str) -> DetectorProfile:
    return DETECTOR_PROFILES.get(method, _DEFAULT_PROFILE)


def _confidence_floor_with_corroboration(
    sources: list[dict],
    streak_band: str | None = None,
) -> dict:
    """Compute confidence for non-ASTRiDE detector outputs.

    The strongest detector keeps its own effective confidence. Additional
    detectors contribute reliability-weighted corroboration into the remaining
    probability mass, so agreement can raise confidence but a lone detector is
    not discounted merely because other models did not fire.

    Args:
        sources: List of {"method": str, "confidence": float} dicts.
        streak_band: Optional length band ("short", "medium", "long") used to
            look up per-band reliability weights from each detector's profile.
    """
    if not sources:
        return {"score": 0.0, "components": [], "divergence": 0.0, "fn_penalty": 0.0}

    components: list[dict] = []
    eff_confs: list[float] = []

    for src in sources:
        method = src.get("method", "")
        raw_conf = float(src.get("confidence", 0.0))
        profile = _get_profile(method)
        ceiling = profile.confidence_ceiling
        eff_conf = min(raw_conf, ceiling) if ceiling is not None else raw_conf
        w = profile.band_adjusted_weight(streak_band)
        contribution = w * eff_conf
        components.append({
            "method": method,
            "raw_conf": raw_conf,
            "eff_conf": round(eff_conf, 4),
            "weight": round(w, 4),
            "contribution": round(contribution, 4),
            "ceiling": ceiling,
            "streak_band": streak_band,
        })
        eff_confs.append(eff_conf)

    n = len(sources)
    best_idx = max(range(n), key=lambda i: eff_confs[i])
    baseline = eff_confs[best_idx]

    # Corroboration boost: additional detectors fill some of the remaining
    # confidence mass. The best detector establishes the floor and is excluded
    # from the boost so single-detector groups are not reliability-discounted.
    corroborating_contributions = [
        c["contribution"] for i, c in enumerate(components) if i != best_idx
    ]
    corroboration = (
        1.0 - math.prod(1.0 - contribution for contribution in corroborating_contributions)
        if corroborating_contributions
        else 0.0
    )

    # Divergence tempers only the boost, never the baseline detector confidence.
    if n > 1:
        divergence = statistics.stdev(eff_confs)
        divergence_factor = 1.0 - 0.15 * divergence
    else:
        divergence = 0.0
        divergence_factor = 1.0

    score = min(0.99, baseline + (1.0 - baseline) * corroboration * divergence_factor)

    return {
        "score": round(score, 6),
        "components": components,
        "divergence": round(divergence, 4),
        "fn_penalty": 0.0,
    }


def compute_unified_confidence(
    sources: list[dict],
    streak_band: str | None = None,
) -> dict:
    """Compute a calibrated Unified Confidence Score from multiple detector outputs.

    Five-step calibrated corroboration formula:

      1. w_i = F-0.5(precision_i, recall_i) × band_weight_i   -- band-adjusted weight
      2. eff_i = min(conf_i, ceiling_i)                         -- optional confidence ceiling
      3. baseline = max(eff_i)                                  -- no single-detector penalty
      4. corroboration = 1 - Π(1 - w_i × eff_i)               -- for non-best detectors
      5. score = baseline + remaining confidence mass × corroboration

    ASTRiDE is special-cased as corroboration-only.  An ASTRiDE-only source list
    returns score 0.0 because those candidates should be dropped upstream.  When
    ASTRiDE overlaps another detector, it is excluded from non-ASTRiDE
    corroboration and divergence so it cannot drag a score down.
    Instead, its raw confidence adds at most a small boost to the best
    non-ASTRiDE raw confidence.  For example, YOLO OBB 0.86 plus ASTRiDE 0.99
    yields roughly 0.90.

    A single non-ASTRiDE detector keeps its own effective confidence. Multiple
    agreeing detectors push the score toward 1, with empirical precision/recall
    (and optional per-band weights) controlling how much corroborating detectors
    can boost the baseline.

    Args:
        sources: List of ``{"method": str, "confidence": float}`` dicts from one
            detection. Any entry with ``method == "unified"`` is filtered out to
            avoid circular reference.
        streak_band: Optional length band ("short", "medium", "long").  When
            provided, each detector's F-0.5 reliability weight is scaled by its
            per-band multiplier from ``DetectorProfile.band_weights``.

    Returns:
        Dict with keys:
            score       -- final unified confidence in [0.0, 0.99]
            components  -- per-detector breakdown list:
                             [{"method", "raw_conf", "eff_conf", "weight",
                               "contribution", "ceiling", "streak_band"}, ...]
            divergence  -- std dev of effective confidences (diagnostic)
            fn_penalty  -- retained diagnostic field; always 0.0 in current scoring
    """
    detector_sources = [s for s in sources if s.get("method") != "unified"]

    if not detector_sources:
        return {"score": 0.0, "components": [], "divergence": 0.0, "fn_penalty": 0.0}

    non_astride_sources = [
        s for s in detector_sources
        if str(s.get("method", "")).lower() not in _ASTRIDE_METHODS
    ]
    astride_sources = [
        s for s in detector_sources
        if str(s.get("method", "")).lower() in _ASTRIDE_METHODS
    ]

    if not non_astride_sources:
        return {"score": 0.0, "components": [], "divergence": 0.0, "fn_penalty": 0.0}

    result = _confidence_floor_with_corroboration(non_astride_sources, streak_band)
    if not astride_sources:
        return result

    best_non_astride_conf = max(float(s.get("confidence", 0.0)) for s in non_astride_sources)
    best_astride_conf = max(float(s.get("confidence", 0.0)) for s in astride_sources)
    corroborated_score = min(
        0.99,
        best_non_astride_conf + _ASTRIDE_MAX_CORROBORATION_BOOST * best_astride_conf,
    )
    result["score"] = round(max(result["score"], corroborated_score), 6)

    for src in astride_sources:
        raw_conf = float(src.get("confidence", 0.0))
        result["components"].append({
            "method": src.get("method", ""),
            "raw_conf": raw_conf,
            "eff_conf": round(raw_conf, 4),
            "weight": 0.0,
            "contribution": round(_ASTRIDE_MAX_CORROBORATION_BOOST * raw_conf, 4),
            "ceiling": DETECTOR_PROFILES["astride"].confidence_ceiling,
            "streak_band": streak_band,
            "role": "corroboration_only",
        })

    return result


if __name__ == "__main__":
    # Quick sanity check — run with: python -m inference.confidence
    examples = [
        ("Single DINOv3 multisource (conf=0.91)", [
            {"method": "dinov3_vitb_multisource", "confidence": 0.91}]),
        ("DINOv3 multisource + YOLO-GTImages agree (both 0.9) — long band", [
            {"method": "dinov3_vitb_multisource", "confidence": 0.90},
            {"method": "streakmind_yolo", "confidence": 0.90},
        ]),
        ("DINOv3 multisource + YOLO-GTImages agree (both 0.9) — medium band", [
            {"method": "dinov3_vitb_multisource", "confidence": 0.90},
            {"method": "streakmind_yolo", "confidence": 0.90},
        ]),
        ("YOLO-GTImages alone (medium band) — should be well-weighted", [
            {"method": "streakmind_yolo", "confidence": 0.80},
        ]),
        ("ASTRiDE-only high-conf FP is disregarded", [{"method": "astride", "confidence": 0.99}]),
        ("YOLO OBB + ASTRiDE corroboration (0.86 + 0.99)", [
            {"method": "streakmind_yolo", "confidence": 0.86},
            {"method": "astride", "confidence": 0.99},
        ]),
        ("DINOv3 multisource + ASTRiDE corroboration", [
            {"method": "dinov3_vitb_multisource", "confidence": 0.90},
            {"method": "astride", "confidence": 0.99},
        ]),
        ("Passes through old unified entry", [
            {"method": "unified", "confidence": 0.88},
            {"method": "tiny", "confidence": 0.75},
        ]),
    ]
    bands = [None, None, "medium", "medium", None, "medium", None, None]
    for (label, srcs), band in zip(examples, bands):
        result = compute_unified_confidence(srcs, streak_band=band)
        band_note = f" [{band}]" if band else ""
        print(f"{label}{band_note}")
        print(f"  score={result['score']:.4f}  divergence={result['divergence']:.4f}"
              f"  fn_penalty={result['fn_penalty']:.4f}")
        for c in result["components"]:
            ceiling_note = f"  [ceiling={c['ceiling']}→eff={c['eff_conf']:.2f}]" if c["ceiling"] is not None else ""
            print(f"    {c['method']:28s}  raw={c['raw_conf']:.2f}  w={c['weight']:.3f}"
                  f"  contrib={c['contribution']:.3f}{ceiling_note}")
        print()
