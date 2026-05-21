"""inference/confidence.py

Precision-recall calibrated Unified Confidence Score for ARGUS streak detections.

Replaces the plain Noisy-OR formula with an F-beta weighted fusion that:
  1. Weights each detector by its empirical F-0.5 score (precision-heavy)
  2. Caps each detector's effective confidence at an optional ceiling (for
     detectors whose raw confidence magnitude is miscalibrated, e.g. ASTRiDE)
  3. Applies a false-negative adjustment when high-recall detectors are silent
  4. Applies a divergence penalty when detectors strongly disagree
  5. Treats ASTRiDE as corroboration-only: it cannot create a standalone
     streak confidence and can only add a small boost to non-ASTRiDE detections.

The score represents how confident we can be that a detection is a real streak,
accounting for each detector's known reliability profile.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


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
        notes: Data source / confidence level for these estimates.
    """

    name: str
    precision: float
    recall: float
    confidence_ceiling: float | None = None
    notes: str = ""

    @property
    def f_beta_weight(self) -> float:
        """F-0.5 score used as this detector's reliability weight."""
        return fbeta_weight(self.precision, self.recall, beta=0.5)


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
# Detector profiles — sourced from phase 8 benchmark and DINOv3 evaluation.
# Measured values come from results/phase8_benchmark.json.
# Estimated values are marked "est." and should be updated once per-method
# precision/recall are available from eval/benchmark.py.
# ---------------------------------------------------------------------------

DETECTOR_PROFILES: dict[str, DetectorProfile] = {
    # Source: phase8_benchmark.json — measured
    "tiny": DetectorProfile(
        name="DINO Swin-T",
        precision=0.6667,
        recall=0.7333,
        notes="Phase 8 measured",
    ),
    "yolo": DetectorProfile(
        name="YOLO11-OBB",
        precision=0.6316,
        recall=0.4000,
        notes="Phase 8 measured",
    ),
    "streakmind_yolo": DetectorProfile(
        name="StreakMindYOLO",
        precision=0.0748,
        recall=0.1404,
        notes="GTImages local smoke benchmark, real-only checkpoint",
    ),
    # DINOv3 ViT-B: mAP@0.5=0.74 (Phase C²); P/R to be refined after Phase D eval
    "dinov3_vitb": DetectorProfile(
        name="DINOv3 ViT-B",
        precision=0.80,
        recall=0.78,
        notes="est. from mAP@0.5=0.74; update post Phase D",
    ),
    "dinov3_gt_dm_satstreaks": DetectorProfile(
        name="DINOv3 GT+DM+SatStreaks",
        precision=0.80,
        recall=0.78,
        notes="mAP@0.5=0.740 on test.json; same precision/recall est. as dinov3_vitb baseline",
    ),
    # DINOv3 ViT-L: Phase D target (RTX 5070 Ti workstation); conservative estimate
    "dinov3_vitl": DetectorProfile(
        name="DINOv3 ViT-L",
        precision=0.85,
        recall=0.82,
        notes="est. Phase D target; update post Phase D",
    ),
    # DINO Swin-L: estimated pre-Phase D
    "large": DetectorProfile(
        name="DINO Swin-L",
        precision=0.75,
        recall=0.75,
        notes="est. pre-Phase D",
    ),
    # ASTRiDE classical: frequently reports very high confidence on false positives,
    # so confidence magnitude is unreliable.  The ceiling caps its effective
    # contribution regardless of the raw score it emits — we trust that it fired,
    # not how confidently it says so.
    "astride": DetectorProfile(
        name="ASTRiDE",
        precision=0.50,
        recall=0.70,
        confidence_ceiling=0.6,
        notes="est. no direct benchmark yet; ceiling set for miscalibrated FP confidence",
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


def _weighted_noisy_or(sources: list[dict]) -> dict:
    """Compute the weighted Noisy-OR score for non-ASTRiDE detector outputs."""
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
        w = profile.f_beta_weight
        contribution = w * eff_conf
        components.append({
            "method": method,
            "raw_conf": raw_conf,
            "eff_conf": round(eff_conf, 4),
            "weight": round(w, 4),
            "contribution": round(contribution, 4),
            "ceiling": ceiling,
        })
        eff_confs.append(eff_conf)

    # Weighted Noisy-OR (uses effective confidences)
    p_weighted = 1.0 - math.prod(1.0 - c["contribution"] for c in components)

    # False-negative adjustment.
    n = len(sources)
    fn_sum = sum(
        _get_profile(s.get("method", "")).recall
        * max(0.0, 0.5 - eff_confs[i])
        for i, s in enumerate(sources)
    )
    fn_penalty_raw = fn_sum / n
    p_fn_adjusted = p_weighted * (1.0 - 0.2 * fn_penalty_raw)

    # Divergence factor — penalise strong inter-detector disagreement.
    if n > 1:
        divergence = statistics.stdev(eff_confs)
        divergence_factor = 1.0 - 0.15 * divergence
    else:
        divergence = 0.0
        divergence_factor = 1.0

    score = min(0.99, p_fn_adjusted * divergence_factor)

    return {
        "score": round(score, 6),
        "components": components,
        "divergence": round(divergence, 4),
        "fn_penalty": round(fn_penalty_raw, 4),
    }


def compute_unified_confidence(sources: list[dict]) -> dict:
    """Compute a calibrated Unified Confidence Score from multiple detector outputs.

    Five-step weighted Noisy-OR formula:

      1. w_i = F-0.5(precision_i, recall_i)             -- precision-heavy reliability weight
      2. eff_i = min(conf_i, ceiling_i)                  -- optional confidence ceiling
      3. P_weighted = 1 - Π(1 - w_i × eff_i)            -- weighted Noisy-OR combination
      4. fn_penalty: mild downward adjustment when high-recall detectors are silent
      5. divergence_factor: mild penalty when detectors strongly disagree

    ASTRiDE is special-cased as corroboration-only.  An ASTRiDE-only source list
    returns score 0.0 because those candidates should be dropped upstream.  When
    ASTRiDE overlaps another detector, it is excluded from weighted Noisy-OR,
    false-negative adjustment, and divergence so it cannot drag a score down.
    Instead, its raw confidence adds at most a small boost to the best
    non-ASTRiDE raw confidence.  For example, YOLO OBB 0.86 plus ASTRiDE 0.99
    yields roughly 0.90.

    A detector's maximum single-detector contribution equals w_i × ceiling_i
    (or w_i if no ceiling is set).  Multiple agreeing detectors push the score
    toward 1.  Detectors that are absent or below 0.5 temper the score
    proportional to their recall.

    Args:
        sources: List of ``{"method": str, "confidence": float}`` dicts from one
            detection. Any entry with ``method == "unified"`` is filtered out to
            avoid circular reference.

    Returns:
        Dict with keys:
            score       -- final unified confidence in [0.0, 0.99]
            components  -- per-detector breakdown list:
                             [{"method", "raw_conf", "eff_conf", "weight",
                               "contribution", "ceiling"}, ...]
            divergence  -- std dev of effective confidences (diagnostic)
            fn_penalty  -- false-negative adjustment applied (fractional reduction)
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

    result = _weighted_noisy_or(non_astride_sources)
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
            "role": "corroboration_only",
        })

    return result


if __name__ == "__main__":
    # Quick sanity check — run with: python -m inference.confidence
    examples = [
        ("Single DINOv3 ViT-B (conf=0.91)", [{"method": "dinov3_vitb", "confidence": 0.91}]),
        ("DINOv3 ViT-B + YOLO agree (both 0.9)", [
            {"method": "dinov3_vitb", "confidence": 0.90},
            {"method": "yolo", "confidence": 0.90},
        ]),
        ("DINOv3 ViT-B + YOLO disagree (0.9 vs 0.15)", [
            {"method": "dinov3_vitb", "confidence": 0.90},
            {"method": "yolo", "confidence": 0.15},
        ]),
        ("ASTRiDE-only high-conf FP is disregarded", [{"method": "astride", "confidence": 0.99}]),
        ("YOLO OBB + ASTRiDE corroboration (0.86 + 0.99 ≈ 0.90)", [
            {"method": "yolo", "confidence": 0.86},
            {"method": "astride", "confidence": 0.99},
        ]),
        ("DINOv3 ViT-B + ASTRiDE corroboration", [
            {"method": "dinov3_vitb", "confidence": 0.90},
            {"method": "astride", "confidence": 0.99},
        ]),
        ("Passes through old unified entry", [
            {"method": "unified", "confidence": 0.88},
            {"method": "tiny", "confidence": 0.75},
        ]),
    ]
    for label, srcs in examples:
        result = compute_unified_confidence(srcs)
        print(f"{label}")
        print(f"  score={result['score']:.4f}  divergence={result['divergence']:.4f}"
              f"  fn_penalty={result['fn_penalty']:.4f}")
        for c in result["components"]:
            ceiling_note = f"  [ceiling={c['ceiling']}→eff={c['eff_conf']:.2f}]" if c["ceiling"] is not None else ""
            print(f"    {c['method']:15s}  raw={c['raw_conf']:.2f}  w={c['weight']:.3f}"
                  f"  contrib={c['contribution']:.3f}{ceiling_note}")
        print()
