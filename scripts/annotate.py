"""Unified ARGUS streak OBB annotation tool.

Accepts FITS, PNG, or JPEG image directories. Outputs COCO JSON compatible
with merge_annotations.py.

Replaces annotate_streaks.py (DarkMatters) and annotate_frigate_streaks.py
(Frigate). For BrentImages .strk output use annotate_brentimages_streaks.py.

Usage:
    # DarkMatters JPEG positives (loads image list from curated_streak CSV):
    python scripts/annotate.py \\
        --image-dir /Volumes/External/TrainingData/raw/DarkMatters/exports \\
        --output results/darkmatters_eval/streak_annotations.json

    # Frigate processed PNGs:
    python scripts/annotate.py \\
        --image-dir /Volumes/External/TrainingData/raw/frigate/processed \\
        --output data/annotations/frigate_streaks.json \\
        --hough-preset frigate

    # Any FITS directory:
    python scripts/annotate.py \\
        --image-dir /path/to/fits \\
        --output data/annotations/my_dataset.json \\
        --hough-preset brentimages

    # Pre-compute Hough suggestion cache (run once before annotating):
    python scripts/annotate.py --image-dir ... --output ... --precompute

Keybindings:
    Y / Enter           — accept all Hough suggestions for this image
    Escape              — dismiss suggestions / cancel pending click
    Right / D / Space   — next image
    Left  / A           — previous image
    B                   — mark as blank (no streak), advance to next
    H / T               — toggle Hough hints on/off (global, persists across images)
    Delete / BackSpace  — delete selected OBB
    S                   — save now
    Q                   — quit and save
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_WIDTH = 10

# ---- geometry ----------------------------------------------------------------

def obb_corners(
    cx: float, cy: float, w: float, h: float, angle_deg: float
) -> list[tuple[float, float]]:
    """Return 4 corners of an OBB as (x, y) tuples."""
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
    """Convert two streak endpoints + width to an OBB dict."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    angle_deg = math.degrees(math.atan2(dy, dx))
    return {"cx": cx, "cy": cy, "w": length, "h": width, "angle_deg": angle_deg}


def obb_to_bbox(obb: dict) -> list[float]:
    """Axis-aligned bounding box [x, y, w, h] from OBB."""
    corners = obb_corners(obb["cx"], obb["cy"], obb["w"], obb["h"], obb["angle_deg"])
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]


def point_to_obb_dist(px: float, py: float, obb: dict) -> float:
    return math.hypot(px - obb["cx"], py - obb["cy"])


# ---- FITS loading ------------------------------------------------------------

def fits_to_pil(fits_path: pathlib.Path) -> Image.Image:
    """Load a 16-bit FITS file and return a percentile-stretched RGB PIL Image."""
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


def _is_fits(path: pathlib.Path) -> bool:
    return path.suffix.lower() in {".fits", ".fit", ".fts"}


def load_image(path: pathlib.Path) -> Image.Image:
    """Load any supported image: FITS (autostretch) or PIL-readable (PNG/JPEG)."""
    if _is_fits(path):
        return fits_to_pil(path)
    return Image.open(path).convert("RGB")


# ---- Hough detection ---------------------------------------------------------

# Preset parameters: (threshold, minLineLength, maxLineGap, downsample)
_HOUGH_PRESETS: dict[str, dict] = {
    "darkmatters": {"threshold": 80, "min_length_frac": 0.05, "max_gap": 25, "downsample": 1},
    "frigate":     {"threshold": 35, "min_length_frac": None, "min_length_px": 40, "max_gap": 10, "downsample": 1},
    "brentimages": {"threshold": 60, "min_length_frac": None, "min_length_px": 50, "max_gap": 8,  "downsample": 4},
}


def estimate_streak_width(
    img: np.ndarray, cx: float, cy: float, angle_deg: float, sample_range: int = 60
) -> float:
    """Estimate streak FWHM via perpendicular brightness profile."""
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
        return 20.0
    arr = np.array(profile)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 5.0:
        return 20.0
    normed = (arr - lo) / (hi - lo)
    above = np.where(normed >= 0.5)[0]
    if len(above) < 2:
        return 20.0
    return max(6.0, min(80.0, float(above[-1] - above[0]) * 1.3))


def _deduplicate_obbs(
    obbs: list[dict], angle_thresh: float = 12.0, dist_thresh: float = 60.0
) -> list[dict]:
    kept: list[dict] = []
    for obb in obbs:
        dup = False
        for k in kept:
            diff = abs(obb["angle_deg"] - k["angle_deg"]) % 180
            if diff > 90:
                diff = 180 - diff
            if diff < angle_thresh and math.hypot(obb["cx"] - k["cx"], obb["cy"] - k["cy"]) < dist_thresh:
                dup = True
                break
        if not dup:
            kept.append(obb)
    return kept


def hough_detect_streaks(img_path: pathlib.Path, preset: str = "darkmatters") -> list[dict]:
    """Run Hough line detection on one image. Returns a list of OBB dicts."""
    try:
        import cv2 as cv
    except ImportError:
        log.warning("opencv-python not available — skipping Hough detection")
        return []

    params = _HOUGH_PRESETS.get(preset, _HOUGH_PRESETS["darkmatters"])
    downsample = params.get("downsample", 1)

    pil = load_image(img_path)
    if downsample > 1:
        small = pil.convert("L").resize(
            (pil.width // downsample, pil.height // downsample), Image.LANCZOS
        )
        img = np.array(small)
    else:
        img = np.array(pil.convert("L"))

    h, w = img.shape
    border = 40

    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(img)
    blurred = cv.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv.Canny(blurred, 30, 100)

    threshold = params["threshold"]
    if params.get("min_length_frac"):
        min_len = int(min(w, h) * params["min_length_frac"])
    else:
        min_len = params.get("min_length_px", 40)
    max_gap = params["max_gap"]

    lines = cv.HoughLinesP(edges, 1, np.pi / 360, threshold=threshold,
                           minLineLength=min_len, maxLineGap=max_gap)
    if lines is None:
        return []

    _MAX_WIDTH = 80.0
    obbs: list[dict] = []
    for line in lines:
        x1, y1, x2, y2 = [float(v) * downsample for v in line[0]]
        cx_l = (x1 + x2) / 2
        cy_l = (y1 + y2) / 2
        fw, fh = pil.width, pil.height
        if cx_l < border or cx_l > fw - border or cy_l < border or cy_l > fh - border:
            continue

        angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        ds_cx, ds_cy = cx_l / downsample, cy_l / downsample
        width = estimate_streak_width(img, ds_cx, ds_cy, angle_deg) * downsample

        if width >= _MAX_WIDTH * downsample * 0.95:
            continue

        obb = endpoints_to_obb(x1, y1, x2, y2, max(6.0, width))
        obbs.append({k: round(v, 2) for k, v in obb.items()})

    obbs.sort(key=lambda o: -o["w"])
    return _deduplicate_obbs(obbs)[:5]


def precompute_all_suggestions(
    images: list[dict],
    suggestions_path: pathlib.Path,
    preset: str = "darkmatters",
    force: bool = False,
) -> dict[str, list[dict]]:
    """Hough-detect streaks on all images and write a sidecar JSON cache."""
    cache: dict[str, list[dict]] = {}
    if suggestions_path.exists() and not force:
        with open(suggestions_path) as fh:
            cache = json.load(fh)
        log.info("Loaded %d cached entries from %s", len(cache), suggestions_path.name)

    todo = [e for e in images if str(e["path"]) not in cache]
    log.info("%d images to process (%d already cached)", len(todo), len(cache))

    for i, entry in enumerate(todo):
        log.info("[%d/%d] %s", i + 1, len(todo), entry["path"].name)
        cache[str(entry["path"])] = hough_detect_streaks(entry["path"], preset=preset)
        if (i + 1) % 10 == 0:
            suggestions_path.parent.mkdir(parents=True, exist_ok=True)
            suggestions_path.write_text(json.dumps(cache, indent=2))

    suggestions_path.parent.mkdir(parents=True, exist_ok=True)
    suggestions_path.write_text(json.dumps(cache, indent=2))
    n_hit = sum(1 for v in cache.values() if v)
    log.info("Done. %d/%d images have ≥1 detection. Saved to %s",
             n_hit, len(cache), suggestions_path)
    return cache


# ---- image list loading ------------------------------------------------------

def _win_preview_to_path(win_path: str, dataset_dir: pathlib.Path) -> pathlib.Path | None:
    """Convert a Windows-style preview path from DarkMatters CSV to a local path."""
    m = re.search(r"(set_\d+[/\\]previews[/\\].+\.(?:jpg|jpeg|png))", win_path, re.IGNORECASE)
    if not m:
        return None
    return dataset_dir / m.group(1).replace("\\", "/")


def load_from_csv(dataset_dir: pathlib.Path) -> list[dict]:
    """Load DarkMatters streak-positive images from the latest curated_streak CSV."""
    candidates = sorted(dataset_dir.glob("curated_streak_v*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No curated_streak_v*.csv in {dataset_dir}")
    entries: list[dict] = []
    with open(candidates[-1], newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row.get("source_group", "").startswith("positive"):
                continue
            path = _win_preview_to_path(row["preview_path"], dataset_dir)
            if path and path.exists():
                entries.append({
                    "path": path,
                    "set_id": row.get("set_id", ""),
                    "frame_id": row.get("frame_id", ""),
                })
    return entries


def load_from_dir(image_dir: pathlib.Path, priority_list: pathlib.Path | None = None,
                  min_score: float = 0.0) -> list[dict]:
    """Load all images from a directory (FITS, PNG, or JPEG)."""
    extensions = {".fits", ".fit", ".fts", ".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)
    if not paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    entries = [{"path": p, "score": 0.0} for p in paths]

    if priority_list and priority_list.exists():
        with open(priority_list) as fh:
            pdata = json.load(fh)
        score_map = {e["frame"]: e.get("score", 0.0) for e in pdata.get("frames", [])}
        ranked, unranked = [], []
        for e in entries:
            s = score_map.get(e["path"].name, None)
            if s is not None and s >= min_score:
                e["score"] = s
                ranked.append(e)
            else:
                unranked.append(e)
        ranked.sort(key=lambda e: -e["score"])
        entries = ranked + unranked

    return entries


def load_existing_annotations(output_path: pathlib.Path, source_name: str = "") -> dict[str, Any]:
    if output_path.exists():
        with open(output_path) as fh:
            return json.load(fh)
    return {
        "info": {
            "description": f"ARGUS streak OBB annotations — {source_name}",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "satellite_streak", "supercategory": "streak"}],
        "images": [],
        "annotations": [],
    }


def save_annotations(coco: dict[str, Any], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(coco, indent=2))
    tmp.replace(output_path)


# ---- main application --------------------------------------------------------

class AnnotationApp(tk.Tk):
    def __init__(
        self,
        images: list[dict],
        coco: dict[str, Any],
        output_path: pathlib.Path,
        suggestions: dict[str, list[dict]] | None = None,
        source_name: str = "manual",
        demo_dir: pathlib.Path | None = None,
    ) -> None:
        super().__init__()
        self.title("ARGUS Streak Annotator")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)

        self.images = images
        self.coco = coco
        self.output_path = output_path
        self.suggestions = suggestions or {}
        self.source_name = source_name
        self.demo_dir = demo_dir or pathlib.Path.home() / "Desktop" / "Demo Images"
        self.idx = 0

        # zoom / pan
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start: tuple[int, int] | None = None

        # per-image state
        self._pending_a: tuple[float, float] | None = None
        self._selected_obb_idx: int | None = None
        self._img_obbs: list[dict] = []
        self._img_suggestions: list[dict] = []

        # global state (persists across images)
        self._show_suggestions: bool = True

        # display cache
        self._photo: ImageTk.PhotoImage | None = None
        self._pil_img: Image.Image | None = None

        self._build_ui()
        self._bind_keys()
        self._load_image(0)

    # ---- UI ------------------------------------------------------------------

    def _build_ui(self) -> None:
        ttk.Style(self).theme_use("clam")

        top = tk.Frame(self, bg="#1a1a2e")
        top.pack(fill="x", padx=6, pady=4)

        self.lbl_progress = tk.Label(
            top, text="", fg="#aaaacc", bg="#1a1a2e", font=("Helvetica", 11)
        )
        self.lbl_progress.pack(side="left")

        # Jump-to-index entry
        tk.Label(top, text="  Go to:", fg="#778899", bg="#1a1a2e",
                 font=("Helvetica", 10)).pack(side="left", padx=(20, 2))
        self._goto_var = tk.StringVar()
        goto_entry = tk.Entry(
            top, textvariable=self._goto_var, width=6,
            bg="#2a2a4a", fg="#ccccee", insertbackground="#ccccee",
            relief="flat", font=("Helvetica", 10),
        )
        goto_entry.pack(side="left")
        goto_entry.bind("<Return>", lambda _: (self._goto_index(), self.focus_set()))

        self.lbl_hint = tk.Label(
            top, text="", fg="#88ffcc", bg="#1a1a2e", font=("Helvetica", 11, "italic")
        )
        self.lbl_hint.pack(side="left", padx=20)

        self.lbl_score = tk.Label(
            top, text="", fg="#ffaa33", bg="#1a1a2e", font=("Helvetica", 10)
        )
        self.lbl_score.pack(side="left", padx=10)

        self.lbl_fname = tk.Label(
            top, text="", fg="#667799", bg="#1a1a2e", font=("Helvetica", 9)
        )
        self.lbl_fname.pack(side="right")

        canvas_frame = tk.Frame(self, bg="#000010")
        canvas_frame.pack(fill="both", expand=True, padx=6)

        self.canvas = tk.Canvas(
            canvas_frame, width=CANVAS_W, height=CANVAS_H,
            bg="#000010", cursor="crosshair", highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        bot = tk.Frame(self, bg="#1a1a2e")
        bot.pack(fill="x", padx=6, pady=4)

        tk.Label(bot, text="Width (px):", fg="#aaaacc", bg="#1a1a2e").pack(side="left")
        self.width_var = tk.IntVar(value=DEFAULT_WIDTH)
        self.width_var.trace_add("write", lambda *_: self._draw_overlays())
        tk.Scale(
            bot, from_=1, to=200, orient="horizontal",
            variable=self.width_var, bg="#1a1a2e", fg="#aaaacc",
            highlightthickness=0, troughcolor="#333355", length=180,
            takefocus=0,
        ).pack(side="left", padx=6)

        for txt, cmd in [
            ("◀  [A]",        self._prev),
            ("[D]  ▶",        self._next),
            ("Accept  [Y]",   self._accept_suggestion),
            ("Blank  [B]",    self._mark_blank),
            ("Dismiss  [Esc]", self._cancel_pending),
            ("Delete  [Del]", self._delete_selected),
            ("Save  [S]",     self._save),
        ]:
            tk.Button(
                bot, text=txt, command=cmd,
                bg="#2a2a4a", fg="#ccccee", relief="flat", padx=7, pady=2,
            ).pack(side="left", padx=3)

        self.btn_hints = tk.Button(
            bot, text="Hints ✓ [H]", command=self._toggle_suggestions,
            bg="#2a2a4a", fg="#ffaa33", relief="flat", padx=7, pady=2,
        )
        self.btn_hints.pack(side="left", padx=3)

        tk.Button(
            bot, text="★ Demo  [I]", command=self._flag_demo,
            bg="#2a2a4a", fg="#ffdd55", relief="flat", padx=7, pady=2,
        ).pack(side="left", padx=3)

        self.lbl_count = tk.Label(
            bot, text="", fg="#66ff88", bg="#1a1a2e", font=("Helvetica", 10, "bold")
        )
        self.lbl_count.pack(side="right", padx=8)

        self.lbl_zoom = tk.Label(
            bot, text="", fg="#778899", bg="#1a1a2e", font=("Helvetica", 9)
        )
        self.lbl_zoom.pack(side="right", padx=4)

    def _bind_keys(self) -> None:
        self.bind("<Right>",     lambda _: self._next())
        self.bind("<d>",         lambda _: self._next())
        self.bind("<space>",     lambda _: self._next())
        self.bind("<Left>",      lambda _: self._prev())
        self.bind("<a>",         lambda _: self._prev())
        self.bind("<Delete>",    lambda _: self._delete_selected())
        self.bind("<BackSpace>", lambda _: self._delete_selected())
        self.bind("<Escape>",    lambda _: self._cancel_pending())
        self.bind("<y>",         lambda _: self._accept_suggestion())
        self.bind("<Return>",    lambda _: self._accept_suggestion())
        self.bind("<b>",         lambda _: self._mark_blank())
        self.bind("<s>",         lambda _: self._save())
        self.bind("<q>",         lambda _: self._quit())
        self.bind("<h>",         lambda _: self._toggle_suggestions())
        self.bind("<t>",         lambda _: self._toggle_suggestions())
        self.bind("<i>",         lambda _: self._flag_demo())

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

    # ---- image loading -------------------------------------------------------

    def _load_image(self, idx: int) -> None:
        self.idx = max(0, min(idx, len(self.images) - 1))
        entry = self.images[self.idx]

        try:
            self._pil_img = load_image(entry["path"])
        except Exception as exc:
            log.error("Cannot open %s: %s", entry["path"], exc)
            self._pil_img = Image.new("RGB", (1200, 800), (30, 30, 50))

        self._photo = None

        cw = self.canvas.winfo_width() or CANVAS_W
        ch = self.canvas.winfo_height() or CANVAS_H
        self.zoom = min(cw / self._pil_img.width, ch / self._pil_img.height)
        self.pan_x = (cw - self._pil_img.width  * self.zoom) / 2
        self.pan_y = (ch - self._pil_img.height * self.zoom) / 2

        img_id = self._get_or_create_image_id(entry)
        self._img_obbs = [
            ann["obb"] for ann in self.coco["annotations"]
            if ann["image_id"] == img_id
        ]

        key = str(entry["path"])
        self._img_suggestions = list(self.suggestions.get(key, []))

        self._pending_a = None
        self._selected_obb_idx = None

        self._redraw()
        self._update_labels()

    def _is_blank(self, img_id: int) -> bool:
        for img in self.coco["images"]:
            if img["id"] == img_id:
                return img.get("blank", False)
        return False

    def _get_or_create_image_id(self, entry: dict) -> int:
        fname = str(entry["path"])
        for img in self.coco["images"]:
            if img["file_name"] == fname:
                return img["id"]
        new_id = max((i["id"] for i in self.coco["images"]), default=0) + 1
        self.coco["images"].append({
            "id": new_id,
            "file_name": fname,
            "width":  self._pil_img.width  if self._pil_img else 0,
            "height": self._pil_img.height if self._pil_img else 0,
            "date_captured": datetime.now(timezone.utc).isoformat(),
            "source": self.source_name,
            "set_id":    entry.get("set_id", ""),
            "frame_id":  entry.get("frame_id", ""),
            "screen_score": entry.get("score", 0.0),
        })
        return new_id

    # ---- drawing -------------------------------------------------------------

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._pil_img is None:
            return

        cw = self.canvas.winfo_width() or CANVAS_W
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

            img_id = self._get_or_create_image_id(self.images[self.idx])
            if self._is_blank(img_id):
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

    def _draw_overlays(self) -> None:
        self.canvas.delete("suggestion")
        self.canvas.delete("confirmed")
        self.canvas.delete("pending")
        self.canvas.delete("blank_label")

        img_id = self._get_or_create_image_id(self.images[self.idx])
        if self._is_blank(img_id):
            cw = self.canvas.winfo_width() or CANVAS_W
            self.canvas.create_text(
                cw // 2, 30, text="✗  CONFIRMED BLANK (no streak)",
                fill=BLANK_COLOR, font=("Helvetica", 16, "bold"), tags="blank_label",
            )
            return

        if self._show_suggestions:
            for i, obb in enumerate(self._img_suggestions):
                corners = obb_corners(obb["cx"], obb["cy"], obb["w"], obb["h"], obb["angle_deg"])
                pts = self._corners_to_canvas(corners)
                self.canvas.create_polygon(
                    pts, outline=SUGGESTION_COLOR, fill="", width=1,
                    dash=(8, 4), tags="suggestion",
                )
                cx, cy = self._img_to_canvas(obb["cx"], obb["cy"])
                self.canvas.create_text(
                    cx, cy - 10,
                    text=f"? {i + 1}  w={obb['h']:.0f}px",
                    fill=SUGGESTION_COLOR, font=("Helvetica", 8), tags="suggestion",
                )

        for i, obb in enumerate(self._img_obbs):
            color = SEL_COLOR if i == self._selected_obb_idx else COLORS[i % len(COLORS)]
            corners = obb_corners(obb["cx"], obb["cy"], obb["w"], obb["h"], obb["angle_deg"])
            pts = self._corners_to_canvas(corners)
            self.canvas.create_polygon(
                pts, outline=color, fill="", width=OBB_LINE_WIDTH, tags="confirmed"
            )
            cx, cy = self._img_to_canvas(obb["cx"], obb["cy"])
            r = 4
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    outline=color, fill=color, tags="confirmed")
            self.canvas.create_text(cx + 8, cy - 8, text=f"#{i + 1}",
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
        n = len(self.images)
        annotated_ids = {a["image_id"] for a in self.coco["annotations"]}
        blank_ids = {img["id"] for img in self.coco["images"] if img.get("blank")}
        done = len(annotated_ids | blank_ids)
        self.lbl_progress.config(text=f"Image {self.idx + 1} / {n}  |  Done: {done}")
        self.lbl_fname.config(text=self.images[self.idx]["path"].name[:70])

        score = self.images[self.idx].get("score", 0.0)
        self.lbl_score.config(text=f"score: {score:.2f}" if score else "")

        img_id = self._get_or_create_image_id(self.images[self.idx])
        cur_blank = self._is_blank(img_id)
        count_text  = "BLANK" if cur_blank else f"OBBs: {len(self._img_obbs)}"
        count_color = BLANK_COLOR if cur_blank else "#66ff88"
        self.lbl_count.config(text=count_text, fg=count_color)
        self.lbl_zoom.config(text=f"zoom {self.zoom:.2f}×")

        hints_label = "Hints ✓ [H]" if self._show_suggestions else "Hints  [H]"
        self.btn_hints.config(text=hints_label)

        if cur_blank:
            hint, color = "BLANK — press B to undo", BLANK_COLOR
        elif self._pending_a is not None:
            hint, color = "→ Click streak END", "#88ffcc"
        elif self._img_suggestions and self._show_suggestions:
            hint = f"Y = accept {len(self._img_suggestions)} suggestion(s)  |  click to annotate manually"
            color = SUGGESTION_COLOR
        else:
            hint = "Click streak START  |  B = blank  |  H = toggle hints"
            color = "#88ffcc"
        self.lbl_hint.config(text=hint, fg=color)

    # ---- navigation ----------------------------------------------------------

    def _next(self) -> None:
        self._autosave()
        if self.idx < len(self.images) - 1:
            self._load_image(self.idx + 1)

    def _prev(self) -> None:
        self._autosave()
        if self.idx > 0:
            self._load_image(self.idx - 1)

    def _goto_index(self) -> None:
        """Jump to a 1-based image number entered in the Go-to field."""
        raw = self._goto_var.get().strip()
        if not raw:
            return
        try:
            n = int(raw)
        except ValueError:
            return
        self._autosave()
        self._load_image(n - 1)  # convert 1-based display to 0-based
        self._goto_var.set("")

    # ---- annotation actions --------------------------------------------------

    def _accept_suggestion(self) -> None:
        if not self._img_suggestions:
            return
        img_id = self._get_or_create_image_id(self.images[self.idx])
        if self._is_blank(img_id):
            return
        for obb in self._img_suggestions:
            self._img_obbs.append(obb)
        self.width_var.set(int(round(self._img_suggestions[0]["h"])))
        self._img_suggestions = []
        self._commit_obbs()
        self._draw_overlays()
        self._update_labels()

    def _mark_blank(self) -> None:
        """Toggle blank (no streak) status for the current image and advance."""
        entry = self.images[self.idx]
        img_id = self._get_or_create_image_id(entry)
        currently_blank = self._is_blank(img_id)

        # Clear any OBBs
        self.coco["annotations"] = [
            a for a in self.coco["annotations"] if a["image_id"] != img_id
        ]
        self._img_obbs = []

        for img in self.coco["images"]:
            if img["id"] == img_id:
                img["blank"] = not currently_blank
                break

        self._autosave()
        if not currently_blank:
            # Newly blanked — advance to next
            self._next()
        else:
            # Un-blanked — stay and redraw
            self._redraw()
            self._update_labels()

    def _toggle_suggestions(self) -> None:
        """Toggle Hough hint overlay globally (persists across images)."""
        self._show_suggestions = not self._show_suggestions
        self._draw_overlays()
        self._update_labels()

    def _flag_demo(self) -> None:
        """Copy the current image to the Demo Images folder."""
        import shutil
        src = self.images[self.idx]["path"]
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        dest = self.demo_dir / src.name
        # Append a counter suffix if a file with the same name already exists
        if dest.exists() and dest.resolve() != src.resolve():
            stem, suffix = src.stem, src.suffix
            counter = 1
            while dest.exists():
                dest = self.demo_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(src, dest)
        log.info("Demo copy: %s → %s", src.name, dest)
        # Flash confirmation in the hint label
        orig_text = self.lbl_hint.cget("text")
        orig_color = self.lbl_hint.cget("fg")
        self.lbl_hint.config(text=f"★ Copied to Demo Images: {dest.name}", fg="#ffdd55")
        self.after(1800, lambda: self.lbl_hint.config(text=orig_text, fg=orig_color))

    def _commit_obbs(self) -> None:
        entry = self.images[self.idx]
        img_id = self._get_or_create_image_id(entry)
        self.coco["annotations"] = [
            a for a in self.coco["annotations"] if a["image_id"] != img_id
        ]
        max_id = max((a["id"] for a in self.coco["annotations"]), default=0)
        for obb in self._img_obbs:
            bbox = obb_to_bbox(obb)
            corners = obb_corners(obb["cx"], obb["cy"], obb["w"], obb["h"], obb["angle_deg"])
            seg = [coord for c in corners for coord in c]
            max_id += 1
            self.coco["annotations"].append({
                "id": max_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": round(obb["w"] * obb["h"], 1),
                "iscrowd": 0,
                "segmentation": [seg],
                "obb": obb,
                "attributes": {"source": self.source_name},
            })

    def _delete_selected(self) -> None:
        if self._selected_obb_idx is None or not self._img_obbs:
            return
        self._img_obbs.pop(self._selected_obb_idx)
        self._selected_obb_idx = None
        self._commit_obbs()
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

    def _on_click(self, event: tk.Event) -> None:
        img_id = self._get_or_create_image_id(self.images[self.idx])
        if self._is_blank(img_id):
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        img_w = self._pil_img.width  if self._pil_img else 3000
        img_h = self._pil_img.height if self._pil_img else 2000
        ix = max(0.0, min(float(img_w), ix))
        iy = max(0.0, min(float(img_h), iy))

        if self._pending_a is None:
            for i, obb in enumerate(self._img_obbs):
                if point_to_obb_dist(ix, iy, obb) < 30 / self.zoom:
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
                self._commit_obbs()
                self._pending_a = None
                self._selected_obb_idx = len(self._img_obbs) - 1

        self._draw_overlays()
        self._update_labels()

    def _on_scroll(self, event: tk.Event) -> None:
        if event.num == 4 or event.delta > 0:
            factor = 1.15
        elif event.num == 5 or event.delta < 0:
            factor = 1 / 1.15
        else:
            return
        old = self.zoom
        self.zoom = max(0.05, min(10.0, self.zoom * factor))
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

    # ---- persistence ---------------------------------------------------------

    def _autosave(self) -> None:
        save_annotations(self.coco, self.output_path)

    def _save(self) -> None:
        self._autosave()
        orig = self.lbl_progress.cget("fg")
        self.lbl_progress.config(fg="#00ff88", text="Saved ✓")
        self.after(800, lambda: (
            self.lbl_progress.config(fg=orig),
            self._update_labels(),
        ))

    def _quit(self) -> None:
        self._autosave()
        self.destroy()


# ---- entry point -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--image-dir", type=pathlib.Path, required=True,
        help="Directory containing images (FITS, PNG, or JPEG). "
             "For DarkMatters, pass the exports/ directory — images are loaded from curated_streak CSV.",
    )
    parser.add_argument(
        "--output", type=pathlib.Path,
        default=pathlib.Path("data/annotations/streak_annotations.json"),
        help="Output COCO JSON path (created fresh or resumed).",
    )
    parser.add_argument(
        "--hough-preset", choices=list(_HOUGH_PRESETS), default="darkmatters",
        help="Hough parameter preset (default: darkmatters). "
             "frigate = short streaks (PNG), brentimages = long streaks (FITS 4× downsample).",
    )
    parser.add_argument(
        "--source-name", default="",
        help="Source label written into annotation attributes (e.g. darkmatters_manual).",
    )
    parser.add_argument(
        "--priority-list", type=pathlib.Path, default=None,
        metavar="SCREEN_JSON",
        help="Frigate screener JSON — frames sorted by descending score.",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Skip frames below this score when --priority-list is given.",
    )
    parser.add_argument(
        "--start-at", type=int, default=0,
        help="1-based image number to start at (default: first unannotated).",
    )
    parser.add_argument(
        "--precompute", action="store_true",
        help="Run Hough detection on all images and write suggestion cache, then exit.",
    )
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Ignore existing suggestion cache and reprocess all images.",
    )
    parser.add_argument(
        "--no-suggestions", action="store_true",
        help="Disable Hough hints; annotate fully manually.",
    )
    parser.add_argument(
        "--demo-dir", type=pathlib.Path,
        default=pathlib.Path.home() / "Desktop" / "Demo Images",
        help="Folder where flagged demo images are copied (default: ~/Desktop/Demo Images).",
    )
    args = parser.parse_args()

    suggestions_path = args.output.parent / (args.output.stem + ".suggestions.json")

    # Detect DarkMatters CSV layout vs plain directory
    log.info("Loading image list from %s", args.image_dir)
    csv_files = list(args.image_dir.glob("curated_streak_v*.csv"))
    if csv_files:
        try:
            images = load_from_csv(args.image_dir)
            log.info("Loaded %d positive images from curated_streak CSV", len(images))
        except FileNotFoundError as exc:
            log.error("%s", exc)
            sys.exit(1)
    else:
        try:
            images = load_from_dir(args.image_dir, args.priority_list, args.min_score)
            log.info("Loaded %d images from directory", len(images))
        except FileNotFoundError as exc:
            log.error("%s", exc)
            sys.exit(1)

    if not images:
        log.error("No images found in %s", args.image_dir)
        sys.exit(1)

    if args.precompute:
        precompute_all_suggestions(images, suggestions_path,
                                   preset=args.hough_preset,
                                   force=args.force_recompute)
        return

    # Load suggestion cache
    suggestions: dict[str, list[dict]] = {}
    if args.no_suggestions:
        log.info("Suggestions disabled (--no-suggestions)")
    elif suggestions_path.exists():
        with open(suggestions_path) as fh:
            suggestions = json.load(fh)
        n_with = sum(1 for v in suggestions.values() if v)
        log.info("Loaded suggestions: %d images, %d with detections", len(suggestions), n_with)
    else:
        log.info("No suggestion cache — run with --precompute for auto-hints")

    source_name = args.source_name or args.image_dir.name
    coco = load_existing_annotations(args.output, source_name)

    annotated_ids = {a["image_id"] for a in coco["annotations"]}
    blank_ids = {img["id"] for img in coco["images"] if img.get("blank")}
    already_done = len(annotated_ids | blank_ids)
    if already_done:
        log.info("Resuming — %d images already reviewed (%d blanks)",
                 already_done, len(blank_ids))

    # Determine start index (--start-at is 1-based; 0 means auto)
    if args.start_at > 0:
        start_idx = args.start_at - 1
    else:
        # Jump to first unannotated, unskipped image
        done_fnames = {
            img["file_name"] for img in coco["images"]
            if img["id"] in annotated_ids or img.get("blank")
        }
        start_idx = 0
        for i, entry in enumerate(images):
            if str(entry["path"]) not in done_fnames:
                start_idx = i
                break

    app = AnnotationApp(images, coco, args.output,
                        suggestions=suggestions, source_name=source_name,
                        demo_dir=args.demo_dir)
    app._load_image(start_idx)
    app.mainloop()

    save_annotations(coco, args.output)
    n_ann = len(coco["annotations"])
    n_img = len({a["image_id"] for a in coco["annotations"]})
    n_blank = sum(1 for img in coco["images"] if img.get("blank"))
    log.info("Done. %d OBBs across %d images, %d blanks → %s",
             n_ann, n_img, n_blank, args.output)


if __name__ == "__main__":
    main()
