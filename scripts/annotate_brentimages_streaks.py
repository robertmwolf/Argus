"""Interactive OBB annotation tool for BrentImages FITS observation nights.

Adapted from scripts/annotate_frigate_streaks.py.

Key differences from the Frigate version:
  - Input is raw 16-bit FITS files (6248×4176); rendered via percentile autostretch.
  - Frames are grouped by satellite pass (NORAD ID) and presented chronologically
    within each pass.
  - Output is written back to the .strk stub files in the night directory:
      Reject=0  with pixel coords  →  annotated streak present
      Reject=-1 with zero coords   →  confirmed no-streak frame
    Frames still at Reject=2 are skipped by convert_gtimages.py until annotated.
  - Hough parameters tuned for long streaks (200–1200 px) in 6248×4176 images,
    run on a 4× downsampled copy for speed.

Workflow::

    # 1. Pre-compute Hough suggestion cache (optional, ~5 s/frame for 300 frames):
    python scripts/annotate_brentimages_streaks.py \\
        --night-dir /Volumes/External/TrainingData/raw/BrentImages/Img_20260515_Atwood \\
        --precompute

    # 2. Open the annotator (resumes from last saved state):
    python scripts/annotate_brentimages_streaks.py \\
        --night-dir /Volumes/External/TrainingData/raw/BrentImages/Img_20260515_Atwood

Keybindings:
    Right / D / Space       — next frame
    Left  / A               — previous frame
    N                       — jump to first unannotated frame in next pass
    P                       — jump to first unannotated frame in previous pass
    Y / Enter               — accept all Hough suggestions for this frame
    B                       — mark frame as confirmed no-streak (Reject=-1)
    Escape                  — dismiss suggestions / cancel pending click
    Delete / BackSpace      — delete selected OBB
    H                       — toggle Hough suggestion overlay
    S                       — save .strk files now
    Q                       — save and quit
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import re
import sys
import tkinter as tk
from datetime import datetime, timezone
from tkinter import ttk
from typing import Any

import numpy as np
from PIL import Image, ImageTk

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---- visual constants --------------------------------------------------------
CANVAS_W = 1600
CANVAS_H = 820

COLORS = ["#00ff88", "#ff6644", "#44aaff", "#ffdd00", "#ff44ff"]
SEL_COLOR = "#ffffff"
SUGGESTION_COLOR = "#ffaa33"
BLANK_COLOR = "#ff4444"
OBB_LINE_WIDTH = 2

# Streak half-width used when writing back to .strk (same as convert_gtimages.py)
_DEFAULT_STREAK_WIDTH = 16.0


# ---- geometry (identical to annotate_frigate_streaks.py) --------------------

def obb_corners(
    cx: float, cy: float, w: float, h: float, angle_deg: float
) -> list[tuple[float, float]]:
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    hw, hh = w / 2, h / 2
    return [
        (cx + cos_a * hw - sin_a * hh, cy + sin_a * hw + cos_a * hh),
        (cx - cos_a * hw - sin_a * hh, cy - sin_a * hw + cos_a * hh),
        (cx - cos_a * hw + sin_a * hh, cy - sin_a * hw - cos_a * hh),
        (cx + cos_a * hw + sin_a * hh, cy + sin_a * hw - cos_a * hh),
    ]


def endpoints_to_obb(
    x1: float, y1: float, x2: float, y2: float, width: float
) -> dict[str, float]:
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    angle_deg = math.degrees(math.atan2(dy, dx))
    return {"cx": cx, "cy": cy, "w": length, "h": width, "angle_deg": angle_deg}


def point_to_obb_dist(px: float, py: float, obb: dict) -> float:
    return math.hypot(px - obb["cx"], py - obb["cy"])


def estimate_streak_width(
    img: np.ndarray, cx: float, cy: float, angle_deg: float, sample_range: int = 80
) -> float:
    """Estimate streak FWHM in the perpendicular direction."""
    perp_rad = math.radians(angle_deg + 90)
    cos_p, sin_p = math.cos(perp_rad), math.sin(perp_rad)
    h, w = img.shape[:2]
    profile = []
    for d in range(-sample_range, sample_range + 1):
        x = int(round(cx + d * cos_p))
        y = int(round(cy + d * sin_p))
        if 0 <= x < w and 0 <= y < h:
            profile.append(float(img[y, x]))
    if len(profile) < 5:
        return _DEFAULT_STREAK_WIDTH
    arr = np.array(profile)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 5.0:
        return _DEFAULT_STREAK_WIDTH
    normed = (arr - lo) / (hi - lo)
    above = np.where(normed >= 0.5)[0]
    if len(above) < 2:
        return _DEFAULT_STREAK_WIDTH
    return max(8.0, min(120.0, float(above[-1] - above[0]) * 1.3))


def _deduplicate_obbs(
    obbs: list[dict], angle_thresh: float = 10.0, dist_thresh: float = 80.0
) -> list[dict]:
    kept: list[dict] = []
    for obb in obbs:
        dup = False
        for k in kept:
            diff = abs(obb["angle_deg"] - k["angle_deg"]) % 180
            if diff > 90:
                diff = 180 - diff
            if diff < angle_thresh and math.hypot(
                obb["cx"] - k["cx"], obb["cy"] - k["cy"]
            ) < dist_thresh:
                dup = True
                break
        if not dup:
            kept.append(obb)
    return kept


# ---- FITS loading & autostretch ---------------------------------------------

def fits_to_pil(fits_path: pathlib.Path) -> Image.Image:
    """Load a 16-bit FITS file and return a percentile-stretched RGB PIL Image.

    Uses a p1/p99 stretch with slight shadow lift for good streak visibility
    without clipping bright stars.
    """
    from astropy.io import fits

    with fits.open(fits_path, memmap=False) as hdul:
        hdul.verify("silentfix")
        data = hdul[0].data.astype(np.float32)

    lo = float(np.percentile(data, 0.5))
    hi = float(np.percentile(data, 99.8))
    if hi <= lo:
        hi = lo + 1.0

    stretched = np.clip((data - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(stretched, mode="L").convert("RGB")


# ---- Hough detection (BrentImages-tuned for long streaks) -------------------

def hough_detect_streaks_fits(fits_path: pathlib.Path, downsample: int = 4) -> list[dict]:
    """Detect long streaks in a FITS file via Hough transform.

    Runs on a downsampled copy for speed; scales coordinates back to full res.
    Tuned for BrentImages 6248×4176 with 200–1200 px streaks.

    Args:
        fits_path: FITS file path.
        downsample: Downsample factor (default 4 → 1562×1044).

    Returns:
        List of OBB dicts in full-resolution pixel coordinates.
    """
    try:
        import cv2 as cv
    except ImportError:
        log.warning("opencv-python not available — skipping Hough detection")
        return []

    pil = fits_to_pil(fits_path)
    small = pil.convert("L").resize(
        (pil.width // downsample, pil.height // downsample), Image.LANCZOS
    )
    img = np.array(small)
    h, w = img.shape

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(img)
    blurred = cv.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv.Canny(blurred, 20, 80)

    lines = cv.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=60,          # 60 votes at 4× downsample ≈ 240 px full-res
        minLineLength=50,      # 50 px at 4× ≈ 200 px full-res
        maxLineGap=8,
    )
    if lines is None:
        return []

    border = 20
    obbs: list[dict] = []
    for line in lines:
        x1, y1, x2, y2 = [float(v) * downsample for v in line[0]]
        cx_l = (x1 + x2) / 2
        cy_l = (y1 + y2) / 2
        fw, fh = pil.width, pil.height
        if cx_l < border or cx_l > fw - border or cy_l < border or cy_l > fh - border:
            continue

        angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Estimate width on the downsampled image
        ds_cx, ds_cy = cx_l / downsample, cy_l / downsample
        width_ds = estimate_streak_width(img, ds_cx, ds_cy, angle_deg, sample_range=40)
        width = width_ds * downsample

        obb = endpoints_to_obb(x1, y1, x2, y2, max(8.0, min(120.0, width)))
        obbs.append({k: round(v, 2) for k, v in obb.items()})

    obbs.sort(key=lambda o: -o["w"])
    return _deduplicate_obbs(obbs)[:5]


def precompute_all_suggestions(
    frames: list[dict],
    cache_path: pathlib.Path,
    force: bool = False,
) -> dict[str, list[dict]]:
    """Run Hough detection on all FITS files and write a sidecar JSON cache."""
    cache: dict[str, list[dict]] = {}
    if cache_path.exists() and not force:
        cache = json.loads(cache_path.read_text())
        log.info("Loaded %d cached entries from %s", len(cache), cache_path.name)

    todo = [f for f in frames if str(f["fits_path"]) not in cache]
    log.info("%d frames to process (%d already cached)", len(todo), len(cache))

    for i, frame in enumerate(todo):
        log.info("[%d/%d] %s", i + 1, len(todo), frame["fits_path"].name)
        cache[str(frame["fits_path"])] = hough_detect_streaks_fits(frame["fits_path"])
        if (i + 1) % 10 == 0:
            cache_path.write_text(json.dumps(cache, indent=2))
    cache_path.write_text(json.dumps(cache, indent=2))

    n_hit = sum(1 for v in cache.values() if v)
    log.info("Done. %d/%d frames have ≥1 detection. Saved to %s",
             n_hit, len(cache), cache_path)
    return cache


# ---- .strk parsing & write-back ---------------------------------------------

def load_strk_lines(strk_path: pathlib.Path) -> list[str]:
    return strk_path.read_text(encoding="utf-8").splitlines()


def write_strk_lines(strk_path: pathlib.Path, lines: list[str]) -> None:
    tmp = strk_path.with_suffix(".strk.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(strk_path)


def update_strk_obs(
    lines: list[str],
    filename: str,
    x1: float, y1: float,
    x2: float, y2: float,
    reject: str,
    jd: float = 0.0,
    exposure: float = 0.5,
    gain: float = 0.0,
) -> list[str]:
    """Update (or no-op) the OBS row for *filename* in the given line list.

    Computes midpoint, length from the endpoints. All SNR/RA/Dec fields are
    zeroed — they will be filled in if the image is plate-solved later.

    Args:
        lines: Raw .strk file lines (mutated in place copy returned).
        filename: Basename of the FITS file to update.
        x1, y1: Streak start pixel (image coords).
        x2, y2: Streak end pixel (image coords).
        reject: "0" = streak present, "-1" = no streak.
        jd: Julian date midpoint (from DATE-OBS, 0 if unknown).
        exposure: Exposure time in seconds.
        gain: Camera gain value.

    Returns:
        New line list with the updated OBS row.
    """
    dx, dy = x2 - x1, y2 - y1
    length = round(math.hypot(dx, dy), 1)
    cx = round((x1 + x2) / 2, 1)
    cy = round((y1 + y2) / 2, 1)
    elongation = round(math.degrees(math.atan2(abs(dy), abs(dx))), 1) if length > 0 else 0.0

    new_fields = [
        filename,
        "",                        # DATE-OBS — preserved from original below
        f"{jd:.10f}",
        f"{x1:.0f}", f"{y1:.0f}",
        f"{x2:.0f}", f"{y2:.0f}",
        f"{cx:.1f}", f"{cy:.1f}",
        "0", "0",                  # Peak SNR, Mean SNR — unknown
        f"{elongation:.1f}",
        f"{length:.1f}",
        reject,
        "0", "0",                  # Mid RA, Dec
        "0", "0",                  # Start RA, Dec
        "0", "0",                  # End RA, Dec
        "0", "0",                  # Expected RA, Dec
        "0",                       # Expected Range
        f"{exposure:.4f}",
        f"{gain:.1f}",
        "Manual annotation" if reject == "0" else "Confirmed no streak",
    ]

    result = []
    in_obs = False
    for line in lines:
        stripped = line.strip()
        if stripped == "[OBS]":
            in_obs = True
            result.append(line)
            continue
        if in_obs and stripped and not stripped.startswith("["):
            parts = stripped.split("\t")
            if parts[0].strip() == filename:
                # Preserve the original DATE-OBS from this row (column 1)
                orig_date = parts[1].strip() if len(parts) > 1 else ""
                if orig_date:
                    new_fields[1] = orig_date
                    # Recompute JD from DATE-OBS if we don't have one
                    if jd == 0.0 and orig_date:
                        try:
                            from astropy.time import Time
                            new_fields[2] = f"{Time(orig_date, format='isot', scale='utc').jd:.10f}"
                        except Exception:
                            pass
                result.append("\t".join(new_fields))
                continue
        result.append(line)
    return result


# ---- data loading ------------------------------------------------------------

_STRK_FNAME_RE = re.compile(r"Streak_(\d+)_(\d{6})\.fits", re.IGNORECASE)


def load_night(night_dir: pathlib.Path) -> list[dict]:
    """Load all frames from a BrentImages night, grouped by NORAD pass.

    Reads .strk stub files to get satellite names and per-frame metadata.
    Returns a flat list ordered by NORAD ID then DATE-OBS; each entry has:
        fits_path, filename, norad_id, sat_name, strk_path,
        date_obs, jd, exposure, gain,
        reject (current value from .strk),
        pass_idx, frame_in_pass, total_in_pass.
    """
    strk_files = sorted(night_dir.glob("*.strk"))
    if not strk_files:
        raise FileNotFoundError(f"No .strk files found in {night_dir} — run generate_brentimages_strk.py first")

    passes: list[list[dict]] = []

    for strk_path in strk_files:
        lines = load_strk_lines(strk_path)
        norad_id: int | None = None
        sat_name = ""
        in_obs = False
        obs_entries: list[dict] = []

        for line in lines:
            stripped = line.strip()
            if stripped == "[OBS]":
                in_obs = False   # next line is header
                continue
            if stripped.startswith("Image\t"):
                in_obs = True
                continue
            if stripped.startswith("[") or not stripped:
                in_obs = False
                continue

            parts = stripped.split("\t")

            # TLE row
            if (not in_obs and len(parts) >= 16
                    and parts[0].strip().isdigit()
                    and not parts[0].strip().startswith("Image")):
                norad_id = int(parts[0].strip())
                sat_name = parts[15].strip()
                continue

            if in_obs and len(parts) >= 14:
                fname = parts[0].strip()
                fits_path = night_dir / fname
                if not fits_path.exists():
                    continue
                try:
                    obs_entries.append({
                        "fits_path": fits_path,
                        "filename": fname,
                        "norad_id": norad_id,
                        "sat_name": sat_name,
                        "strk_path": strk_path,
                        "date_obs": parts[1].strip(),
                        "jd": float(parts[2]) if parts[2].strip() else 0.0,
                        "x_start": float(parts[3]) if parts[3].strip() else 0.0,
                        "y_start": float(parts[4]) if parts[4].strip() else 0.0,
                        "x_end":   float(parts[5]) if parts[5].strip() else 0.0,
                        "y_end":   float(parts[6]) if parts[6].strip() else 0.0,
                        "reject":  parts[13].strip(),
                        "exposure": float(parts[23].strip()) if len(parts) > 23 and parts[23].strip() else 0.5,
                        "gain":     float(parts[24].strip()) if len(parts) > 24 and parts[24].strip() else 0.0,
                    })
                except (ValueError, IndexError):
                    pass

        if obs_entries and norad_id is not None:
            obs_entries.sort(key=lambda e: e["date_obs"])
            passes.append(obs_entries)

    # Flatten and attach pass/frame counters
    frames: list[dict] = []
    for pass_idx, pass_frames in enumerate(passes):
        total = len(pass_frames)
        for frame_idx, frame in enumerate(pass_frames):
            frame["pass_idx"] = pass_idx
            frame["frame_in_pass"] = frame_idx
            frame["total_in_pass"] = total
            frame["num_passes"] = len(passes)
            frames.append(frame)

    return frames


# ---- main application --------------------------------------------------------

class AnnotationApp(tk.Tk):

    def __init__(
        self,
        frames: list[dict],
        night_dir: pathlib.Path,
        suggestions: dict[str, list[dict]] | None = None,
        start_at: int = 0,
    ) -> None:
        super().__init__()
        self.title("ARGUS BrentImages Streak Annotator")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)

        self.frames = frames
        self.night_dir = night_dir
        self.suggestions = suggestions or {}
        self.idx = 0

        # in-memory .strk line caches: strk_path → list[str]
        self._strk_cache: dict[pathlib.Path, list[str]] = {}

        # zoom / pan
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start: tuple[int, int] | None = None

        # per-frame annotation state
        self._pending_a: tuple[float, float] | None = None
        self._selected_obb_idx: int | None = None
        self._img_obbs: list[dict] = []          # confirmed OBBs for current frame
        self._img_suggestions: list[dict] = []
        self._show_suggestions: bool = bool(suggestions)
        self._is_blank: bool = False             # True if frame marked no-streak

        # display cache
        self._photo: ImageTk.PhotoImage | None = None
        self._pil_img: Image.Image | None = None

        self._build_ui()
        self._bind_keys()
        self.after(50, lambda: self._load_frame(start_at))

    # ---- .strk cache helpers -------------------------------------------------

    def _get_strk_lines(self, strk_path: pathlib.Path) -> list[str]:
        if strk_path not in self._strk_cache:
            self._strk_cache[strk_path] = load_strk_lines(strk_path)
        return self._strk_cache[strk_path]

    def _flush_strk(self, strk_path: pathlib.Path) -> None:
        if strk_path in self._strk_cache:
            write_strk_lines(strk_path, self._strk_cache[strk_path])

    def _flush_all(self) -> None:
        for strk_path in list(self._strk_cache.keys()):
            self._flush_strk(strk_path)

    # ---- UI ------------------------------------------------------------------

    def _build_ui(self) -> None:
        ttk.Style(self).theme_use("clam")

        # ── satellite / pass info bar ──────────────────────────────────────────
        sat_bar = tk.Frame(self, bg="#0d1117")
        sat_bar.pack(fill="x", padx=6, pady=(4, 0))

        self.lbl_sat = tk.Label(
            sat_bar, text="", fg="#44ddff", bg="#0d1117",
            font=("Helvetica", 13, "bold"),
        )
        self.lbl_sat.pack(side="left")

        self.lbl_pass = tk.Label(
            sat_bar, text="", fg="#778899", bg="#0d1117", font=("Helvetica", 10),
        )
        self.lbl_pass.pack(side="left", padx=14)

        self.lbl_date = tk.Label(
            sat_bar, text="", fg="#556677", bg="#0d1117", font=("Helvetica", 9),
        )
        self.lbl_date.pack(side="right")

        # ── top status bar ────────────────────────────────────────────────────
        top = tk.Frame(self, bg="#1a1a2e")
        top.pack(fill="x", padx=6, pady=2)

        self.lbl_progress = tk.Label(
            top, text="", fg="#aaaacc", bg="#1a1a2e", font=("Helvetica", 11),
        )
        self.lbl_progress.pack(side="left")

        self.lbl_hint = tk.Label(
            top, text="", fg="#88ffcc", bg="#1a1a2e", font=("Helvetica", 11, "italic"),
        )
        self.lbl_hint.pack(side="left", padx=20)

        self.lbl_fname = tk.Label(
            top, text="", fg="#667799", bg="#1a1a2e", font=("Helvetica", 9),
        )
        self.lbl_fname.pack(side="right")

        # ── canvas ────────────────────────────────────────────────────────────
        canvas_frame = tk.Frame(self, bg="#000010")
        canvas_frame.pack(fill="both", expand=True, padx=6)

        self.canvas = tk.Canvas(
            canvas_frame, width=CANVAS_W, height=CANVAS_H,
            bg="#000010", cursor="crosshair", highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # ── bottom control bar ────────────────────────────────────────────────
        bot = tk.Frame(self, bg="#1a1a2e")
        bot.pack(fill="x", padx=6, pady=4)

        tk.Label(bot, text="Width (px):", fg="#aaaacc", bg="#1a1a2e").pack(side="left")
        self.width_var = tk.IntVar(value=16)
        self.width_var.trace_add("write", lambda *_: self._draw_overlays())
        tk.Scale(
            bot, from_=1, to=200, orient="horizontal",
            variable=self.width_var, bg="#1a1a2e", fg="#aaaacc",
            highlightthickness=0, troughcolor="#333355", length=180,
        ).pack(side="left", padx=6)

        for txt, cmd in [
            ("◀ [A]",          self._prev),
            ("[D] ▶",          self._next),
            ("Prev pass [P]",  self._prev_pass),
            ("Next pass [N]",  self._next_pass),
            ("Accept [Y]",     self._accept_suggestion),
            ("Blank [B]",      self._mark_blank),
            ("Delete [Del]",   self._delete_selected),
            ("Save [S]",       self._save),
        ]:
            tk.Button(
                bot, text=txt, command=cmd,
                bg="#2a2a4a", fg="#ccccee", relief="flat", padx=6, pady=2,
            ).pack(side="left", padx=2)

        self.btn_hints = tk.Button(
            bot, text="Hints [H]", command=self._toggle_suggestions,
            bg="#2a2a4a", fg="#ffaa33", relief="flat", padx=6, pady=2,
        )
        self.btn_hints.pack(side="left", padx=2)

        self.lbl_count = tk.Label(
            bot, text="", fg="#66ff88", bg="#1a1a2e", font=("Helvetica", 10, "bold"),
        )
        self.lbl_count.pack(side="right", padx=8)

        self.lbl_zoom = tk.Label(
            bot, text="", fg="#778899", bg="#1a1a2e", font=("Helvetica", 9),
        )
        self.lbl_zoom.pack(side="right", padx=4)

    def _bind_keys(self) -> None:
        self.bind("<Right>",     lambda _: self._next())
        self.bind("<d>",         lambda _: self._next())
        self.bind("<space>",     lambda _: self._next())
        self.bind("<Left>",      lambda _: self._prev())
        self.bind("<a>",         lambda _: self._prev())
        self.bind("<n>",         lambda _: self._next_pass())
        self.bind("<p>",         lambda _: self._prev_pass())
        self.bind("<Delete>",    lambda _: self._delete_selected())
        self.bind("<BackSpace>", lambda _: self._delete_selected())
        self.bind("<Escape>",    lambda _: self._cancel_pending())
        self.bind("<y>",         lambda _: self._accept_suggestion())
        self.bind("<Return>",    lambda _: self._accept_suggestion())
        self.bind("<b>",         lambda _: self._mark_blank())
        self.bind("<s>",         lambda _: self._save())
        self.bind("<q>",         lambda _: self._quit())
        self.bind("<h>",         lambda _: self._toggle_suggestions())

        self.canvas.bind("<Button-1>",        self._on_click)
        self.canvas.bind("<Button-2>",        self._on_pan_start)
        self.canvas.bind("<B2-Motion>",       self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.canvas.bind("<Button-3>",        self._on_pan_start)
        self.canvas.bind("<B3-Motion>",       self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>",      self._on_scroll)
        self.canvas.bind("<Button-4>",        self._on_scroll)
        self.canvas.bind("<Button-5>",        self._on_scroll)

    # ---- coordinate helpers --------------------------------------------------

    def _canvas_to_img(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self.pan_x) / self.zoom, (cy - self.pan_y) / self.zoom

    def _img_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        return ix * self.zoom + self.pan_x, iy * self.zoom + self.pan_y

    def _corners_to_canvas(self, corners: list[tuple[float, float]]) -> list[float]:
        flat: list[float] = []
        for ix, iy in corners:
            cx, cy = self._img_to_canvas(ix, iy)
            flat += [cx, cy]
        return flat

    # ---- frame loading -------------------------------------------------------

    def _load_frame(self, idx: int) -> None:
        self.idx = max(0, min(idx, len(self.frames) - 1))
        frame = self.frames[self.idx]

        try:
            self._pil_img = fits_to_pil(frame["fits_path"])
        except Exception as exc:
            log.error("Cannot load %s: %s", frame["fits_path"], exc)
            self._pil_img = Image.new("RGB", (6248, 4176), (20, 20, 35))

        self._photo = None

        # Default zoom: fit to canvas
        img_w, img_h = self._pil_img.width, self._pil_img.height
        self.zoom = min(CANVAS_W / img_w, CANVAS_H / img_h)
        self.pan_x = (CANVAS_W - img_w * self.zoom) / 2
        self.pan_y = (CANVAS_H - img_h * self.zoom) / 2

        # Restore OBBs from existing .strk annotation
        reject = frame["reject"]
        self._is_blank = (reject == "-1")
        self._img_obbs = []
        if reject == "0" and frame["x_end"] != 0.0:
            obb = endpoints_to_obb(
                frame["x_start"], frame["y_start"],
                frame["x_end"],   frame["y_end"],
                _DEFAULT_STREAK_WIDTH,
            )
            self._img_obbs = [obb]

        # Hough suggestions
        key = str(frame["fits_path"])
        self._img_suggestions = list(self.suggestions.get(key, []))
        if self._img_suggestions and not self._img_obbs and not self._is_blank:
            self.width_var.set(int(round(self._img_suggestions[0]["h"])))

        self._pending_a = None
        self._selected_obb_idx = None

        self._redraw()
        self._update_labels()

    # ---- drawing -------------------------------------------------------------

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._pil_img is None:
            return

        cw = self.canvas.winfo_width()  or CANVAS_W
        ch = self.canvas.winfo_height() or CANVAS_H
        img_w, img_h = self._pil_img.width, self._pil_img.height

        vis_x0 = max(0, int(-self.pan_x / self.zoom))
        vis_y0 = max(0, int(-self.pan_y / self.zoom))
        vis_x1 = min(img_w, math.ceil((cw - self.pan_x) / self.zoom) + 1)
        vis_y1 = min(img_h, math.ceil((ch - self.pan_y) / self.zoom) + 1)

        if vis_x1 > vis_x0 and vis_y1 > vis_y0:
            crop = self._pil_img.crop((vis_x0, vis_y0, vis_x1, vis_y1))
            disp_w = max(1, int((vis_x1 - vis_x0) * self.zoom))
            disp_h = max(1, int((vis_y1 - vis_y0) * self.zoom))
            resized = crop.resize((disp_w, disp_h), Image.LANCZOS)

            # Tint blank frames slightly red
            if self._is_blank:
                r, g, b = resized.split()
                r = r.point(lambda v: min(255, int(v * 1.2)))
                g = g.point(lambda v: int(v * 0.7))
                b = b.point(lambda v: int(v * 0.7))
                resized = Image.merge("RGB", (r, g, b))

            self._photo = ImageTk.PhotoImage(resized)
            anchor_x = self.pan_x + vis_x0 * self.zoom
            anchor_y = self.pan_y + vis_y0 * self.zoom
            self.canvas.create_image(anchor_x, anchor_y, anchor="nw",
                                     image=self._photo, tags="bg")

        self._draw_overlays()

    def _toggle_suggestions(self) -> None:
        self._show_suggestions = not self._show_suggestions
        self.btn_hints.config(
            text="Hints ✓ [H]" if self._show_suggestions else "Hints [H]"
        )
        self._draw_overlays()
        self._update_labels()

    def _draw_overlays(self) -> None:
        self.canvas.delete("suggestion")
        self.canvas.delete("confirmed")
        self.canvas.delete("pending")
        self.canvas.delete("blank_label")

        if self._is_blank:
            cw = self.canvas.winfo_width() or CANVAS_W
            self.canvas.create_text(
                cw // 2, 30, text="✗  CONFIRMED NO STREAK",
                fill=BLANK_COLOR, font=("Helvetica", 16, "bold"), tags="blank_label",
            )
            return

        visible_suggestions = self._img_suggestions if self._show_suggestions else []
        for i, obb in enumerate(visible_suggestions):
            corners = obb_corners(obb["cx"], obb["cy"], obb["w"], obb["h"], obb["angle_deg"])
            pts = self._corners_to_canvas(corners)
            self.canvas.create_polygon(
                pts, outline=SUGGESTION_COLOR, fill="", width=1,
                dash=(8, 4), tags="suggestion",
            )
            cx, cy = self._img_to_canvas(obb["cx"], obb["cy"])
            self.canvas.create_text(
                cx, cy - 10, text=f"? {i + 1}  {obb['w']:.0f}px",
                fill=SUGGESTION_COLOR, font=("Helvetica", 8), tags="suggestion",
            )

        for i, obb in enumerate(self._img_obbs):
            color = SEL_COLOR if i == self._selected_obb_idx else COLORS[i % len(COLORS)]
            corners = obb_corners(obb["cx"], obb["cy"], obb["w"], obb["h"], obb["angle_deg"])
            pts = self._corners_to_canvas(corners)
            self.canvas.create_polygon(
                pts, outline=color, fill="", width=OBB_LINE_WIDTH, tags="confirmed",
            )
            cx, cy = self._img_to_canvas(obb["cx"], obb["cy"])
            r = 4
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    outline=color, fill=color, tags="confirmed")
            self.canvas.create_text(cx + 8, cy - 8, text=f"#{i + 1}  {obb['w']:.0f}px",
                                    fill=color, font=("Helvetica", 9), tags="confirmed")

        if self._pending_a is not None:
            cx, cy = self._img_to_canvas(*self._pending_a)
            r = max(3.0, (self.width_var.get() / 2) * self.zoom)
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline="#ffff00", fill="", width=2, dash=(4, 3), tags="pending",
            )
            self.canvas.create_oval(
                cx - 3, cy - 3, cx + 3, cy + 3,
                outline="#ffff00", fill="#ffff00", tags="pending",
            )

    def _update_labels(self) -> None:
        frame = self.frames[self.idx]
        total = len(self.frames)
        n_done = sum(
            1 for f in self.frames if f["reject"] in ("0", "-1")
        )

        # Pass info
        pass_idx = frame["pass_idx"]
        num_passes = frame["num_passes"]
        frame_in = frame["frame_in_pass"]
        total_in = frame["total_in_pass"]
        norad = frame["norad_id"]
        sat = frame["sat_name"]

        self.lbl_sat.config(text=f"NORAD {norad}  —  {sat}")
        self.lbl_pass.config(
            text=f"Pass {pass_idx + 1}/{num_passes}  ·  frame {frame_in + 1}/{total_in}"
        )
        self.lbl_date.config(text=frame["date_obs"])

        self.lbl_progress.config(
            text=f"Frame {self.idx + 1}/{total}  |  Done: {n_done}"
        )
        self.lbl_fname.config(text=frame["filename"])
        self.lbl_count.config(text=f"OBBs: {len(self._img_obbs)}")
        self.lbl_zoom.config(text=f"zoom {self.zoom:.2f}×")

        visible = self._img_suggestions if self._show_suggestions else []
        if self._is_blank:
            hint, color = "Confirmed no streak — press B to undo", BLANK_COLOR
        elif self._pending_a is not None:
            hint, color = "→ Click streak END", "#88ffcc"
        elif visible:
            hint = f"Y = accept {len(visible)} suggestion(s)  |  click to annotate manually"
            color = SUGGESTION_COLOR
        else:
            hint, color = "Click streak START  |  B = no streak in this frame", "#88ffcc"
        self.lbl_hint.config(text=hint, fg=color)

    # ---- interaction ---------------------------------------------------------

    def _on_click(self, event: tk.Event) -> None:
        if self._is_blank:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        img_w = self._pil_img.width  if self._pil_img else 6248
        img_h = self._pil_img.height if self._pil_img else 4176
        ix = max(0.0, min(float(img_w), ix))
        iy = max(0.0, min(float(img_h), iy))

        if self._pending_a is None:
            for i, obb in enumerate(self._img_obbs):
                if point_to_obb_dist(ix, iy, obb) < 40 / self.zoom:
                    self._selected_obb_idx = i
                    self._draw_overlays()
                    self._update_labels()
                    return
            self._selected_obb_idx = None
            self._pending_a = (ix, iy)
        else:
            x1, y1 = self._pending_a
            x2, y2 = ix, iy
            if math.hypot(x2 - x1, y2 - y1) < 5:
                self._pending_a = None
            else:
                obb = endpoints_to_obb(x1, y1, x2, y2, float(self.width_var.get()))
                self._img_obbs.append({k: round(v, 2) for k, v in obb.items()})
                self._commit_annotation()
                self._pending_a = None
                self._selected_obb_idx = len(self._img_obbs) - 1

        self._draw_overlays()
        self._update_labels()

    def _on_scroll(self, event: tk.Event) -> None:
        factor = 1.15 if (event.num == 4 or event.delta > 0) else (
            1 / 1.15 if (event.num == 5 or event.delta < 0) else None
        )
        if factor is None:
            return
        old = self.zoom
        self.zoom = max(0.03, min(12.0, self.zoom * factor))
        self.pan_x = event.x - (event.x - self.pan_x) * (self.zoom / old)
        self.pan_y = event.y - (event.y - self.pan_y) * (self.zoom / old)
        self._redraw()
        self._update_labels()

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start = (event.x, event.y)

    def _on_pan_drag(self, event: tk.Event) -> None:
        if self._pan_start is None:
            return
        self.pan_x += event.x - self._pan_start[0]
        self.pan_y += event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        self._redraw()

    def _on_pan_end(self, _: tk.Event) -> None:
        self._pan_start = None

    # ---- navigation ----------------------------------------------------------

    def _next(self) -> None:
        if self.idx < len(self.frames) - 1:
            self._load_frame(self.idx + 1)

    def _prev(self) -> None:
        if self.idx > 0:
            self._load_frame(self.idx - 1)

    def _next_pass(self) -> None:
        """Jump to first unannotated frame of the next satellite pass."""
        cur_pass = self.frames[self.idx]["pass_idx"]
        target_pass = cur_pass + 1
        for i, f in enumerate(self.frames):
            if f["pass_idx"] == target_pass:
                # Land on the first unannotated frame in that pass, else first
                for j in range(i, len(self.frames)):
                    if self.frames[j]["pass_idx"] != target_pass:
                        break
                    if self.frames[j]["reject"] == "2":
                        self._load_frame(j)
                        return
                self._load_frame(i)
                return

    def _prev_pass(self) -> None:
        """Jump to first unannotated frame of the previous satellite pass."""
        cur_pass = self.frames[self.idx]["pass_idx"]
        target_pass = cur_pass - 1
        if target_pass < 0:
            return
        for i, f in enumerate(self.frames):
            if f["pass_idx"] == target_pass:
                for j in range(i, len(self.frames)):
                    if self.frames[j]["pass_idx"] != target_pass:
                        break
                    if self.frames[j]["reject"] == "2":
                        self._load_frame(j)
                        return
                self._load_frame(i)
                return

    # ---- annotation actions --------------------------------------------------

    def _accept_suggestion(self) -> None:
        if not self._img_suggestions or self._is_blank:
            return
        self._img_obbs = list(self._img_suggestions)
        self.width_var.set(int(round(self._img_suggestions[0]["h"])))
        self._img_suggestions = []
        self._commit_annotation()
        self._draw_overlays()
        self._update_labels()

    def _mark_blank(self) -> None:
        """Toggle confirmed no-streak for this frame."""
        frame = self.frames[self.idx]
        if self._is_blank:
            # Undo: reset to pending
            self._is_blank = False
            self._img_obbs = []
            frame["reject"] = "2"
            frame["x_start"] = frame["y_start"] = 0.0
            frame["x_end"]   = frame["y_end"]   = 0.0
            lines = self._get_strk_lines(frame["strk_path"])
            self._strk_cache[frame["strk_path"]] = update_strk_obs(
                lines, frame["filename"], 0, 0, 0, 0, "2",
                frame["jd"], frame["exposure"], frame["gain"],
            )
        else:
            self._is_blank = True
            self._img_obbs = []
            self._pending_a = None
            frame["reject"] = "-1"
            frame["x_start"] = frame["y_start"] = 0.0
            frame["x_end"]   = frame["y_end"]   = 0.0
            lines = self._get_strk_lines(frame["strk_path"])
            self._strk_cache[frame["strk_path"]] = update_strk_obs(
                lines, frame["filename"], 0, 0, 0, 0, "-1",
                frame["jd"], frame["exposure"], frame["gain"],
            )
        self._redraw()
        self._update_labels()

    def _commit_annotation(self) -> None:
        """Write current OBBs (first one only) to the in-memory .strk cache."""
        frame = self.frames[self.idx]
        if not self._img_obbs:
            return

        obb = self._img_obbs[0]  # .strk stores one streak per frame
        a = math.radians(obb["angle_deg"])
        hw = obb["w"] / 2
        x1 = round(obb["cx"] - math.cos(a) * hw, 1)
        y1 = round(obb["cy"] - math.sin(a) * hw, 1)
        x2 = round(obb["cx"] + math.cos(a) * hw, 1)
        y2 = round(obb["cy"] + math.sin(a) * hw, 1)

        frame["reject"]  = "0"
        frame["x_start"] = x1;  frame["y_start"] = y1
        frame["x_end"]   = x2;  frame["y_end"]   = y2

        lines = self._get_strk_lines(frame["strk_path"])
        self._strk_cache[frame["strk_path"]] = update_strk_obs(
            lines, frame["filename"], x1, y1, x2, y2, "0",
            frame["jd"], frame["exposure"], frame["gain"],
        )

    def _delete_selected(self) -> None:
        if self._selected_obb_idx is None or not self._img_obbs:
            return
        self._img_obbs.pop(self._selected_obb_idx)
        self._selected_obb_idx = None
        if not self._img_obbs:
            # Reset frame to pending
            frame = self.frames[self.idx]
            frame["reject"] = "2"
            lines = self._get_strk_lines(frame["strk_path"])
            self._strk_cache[frame["strk_path"]] = update_strk_obs(
                lines, frame["filename"], 0, 0, 0, 0, "2",
                frame["jd"], frame["exposure"], frame["gain"],
            )
        else:
            self._commit_annotation()
        self._draw_overlays()
        self._update_labels()

    def _cancel_pending(self) -> None:
        if self._pending_a is not None:
            self._pending_a = None
        else:
            self._img_suggestions = []
        self._selected_obb_idx = None
        self._draw_overlays()
        self._update_labels()

    # ---- persistence ---------------------------------------------------------

    def _save(self) -> None:
        self._flush_all()
        orig = self.lbl_progress.cget("fg")
        self.lbl_progress.config(fg="#00ff88", text="Saved ✓")
        self.after(800, lambda: (
            self.lbl_progress.config(fg=orig),
            self._update_labels(),
        ))

    def _quit(self) -> None:
        self._flush_all()
        self.destroy()


# ---- entry point -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--night-dir", required=True, type=pathlib.Path,
        help="Directory containing Streak_NORADID_HHMMSS.fits files and .strk stubs.",
    )
    parser.add_argument(
        "--precompute", action="store_true",
        help="Run Hough detection on all frames and write suggestion cache, then exit.",
    )
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Ignore existing suggestion cache and reprocess all frames.",
    )
    parser.add_argument(
        "--no-suggestions", action="store_true",
        help="Disable Hough hints and annotate fully manually.",
    )
    parser.add_argument(
        "--start-at", type=int, default=0,
        help="0-based frame index to start at.",
    )
    args = parser.parse_args()

    log.info("Loading night from %s", args.night_dir)
    try:
        frames = load_night(args.night_dir)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    log.info("Loaded %d frames across %d passes",
             len(frames),
             max(f["pass_idx"] for f in frames) + 1 if frames else 0)

    cache_path = args.night_dir / "brentimages_suggestions.json"

    if args.precompute:
        precompute_all_suggestions(frames, cache_path, force=args.force_recompute)
        return

    suggestions: dict[str, list[dict]] = {}
    if not args.no_suggestions:
        if cache_path.exists():
            suggestions = json.loads(cache_path.read_text())
            n_hit = sum(1 for v in suggestions.values() if v)
            log.info("Loaded suggestion cache: %d frames, %d with detections", len(suggestions), n_hit)
        else:
            log.info("No suggestion cache — run with --precompute for auto-hints")

    already_done = sum(1 for f in frames if f["reject"] in ("0", "-1"))
    if already_done:
        log.info("Resuming — %d/%d frames already annotated", already_done, len(frames))

    # If resuming, jump to first unannotated frame
    start = args.start_at
    if start == 0 and already_done > 0:
        for i, f in enumerate(frames):
            if f["reject"] == "2":
                start = i
                break

    app = AnnotationApp(frames, args.night_dir, suggestions=suggestions, start_at=start)
    app.mainloop()

    n_done   = sum(1 for f in frames if f["reject"] in ("0", "-1"))
    n_streak = sum(1 for f in frames if f["reject"] == "0")
    n_blank  = sum(1 for f in frames if f["reject"] == "-1")
    n_pending = sum(1 for f in frames if f["reject"] == "2")
    log.info(
        "Session complete. %d/%d frames annotated  "
        "(%d streaks, %d no-streak, %d still pending)",
        n_done, len(frames), n_streak, n_blank, n_pending,
    )


if __name__ == "__main__":
    main()
