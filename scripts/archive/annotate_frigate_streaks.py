"""Interactive OBB annotation tool for Frigate processed PNGs.

Key differences:
  - No CSV dependency — images are loaded by globbing a directory of PNGs.
  - Hough parameters tuned for short streaks (50–150 px) in 2325×1555 images:
      threshold=35  (was 80), minLineLength=40 px (was 77 px), maxLineGap=10 px.
  - Full-width canvas for the image — zoom in to find very short streaks.
  - --priority-list accepts frigate_screen.json from screen_frigate.py; frames
    are presented in descending score order so the most promising are first.
  - Output COCO JSON is compatible with merge_annotations.py (--include-frigate).

Workflow:
  # 1. Pre-compute Hough suggestion cache (optional, ~2 min for 1,980 frames):
  python scripts/annotate_frigate_streaks.py --precompute

  # 2. Annotate, starting from highest-scored frames:
  python scripts/annotate_frigate_streaks.py \\
      --priority-list data/annotations/frigate_screen.json \\
      --min-score 2.0

Keybindings:
    Y / Enter               — accept all suggestions for this image
    Escape                  — dismiss suggestions / cancel pending click
    Right / D / Space       — next image
    Left  / A               — previous image
    Delete / BackSpace      — delete selected OBB
    S                       — save now
    Q                       — quit
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
CANVAS_W = 1600    # full-width single-panel canvas
CANVAS_H = 780

COLORS = ["#00ff88", "#ff6644", "#44aaff", "#ffdd00", "#ff44ff"]
SEL_COLOR = "#ffffff"
SUGGESTION_COLOR = "#ffaa33"
OBB_LINE_WIDTH = 2

# Frigate filename pattern: "Capture_NNNNN HH_MM_SSZ.png"
_FNAME_PAT = re.compile(r"Capture_(\d{5}) (\d{2})_(\d{2})_(\d{2})Z", re.IGNORECASE)


# ---- geometry (reused verbatim from annotate_streaks.py) ---------------------

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


# ---- Hough detection (Frigate-tuned) ----------------------------------------

def estimate_streak_width(
    img: np.ndarray, cx: float, cy: float, angle_deg: float, sample_range: int = 60
) -> float:
    """Estimate streak width as FWHM of the perpendicular brightness profile."""
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

    fwhm = float(above[-1] - above[0])
    return max(6.0, min(80.0, fwhm * 1.3))


def _deduplicate_obbs(
    obbs: list[dict], angle_thresh: float = 12.0, dist_thresh: float = 60.0
) -> list[dict]:
    """Remove near-duplicate detections of the same streak."""
    kept: list[dict] = []
    for obb in obbs:
        dup = False
        for k in kept:
            diff = abs(obb["angle_deg"] - k["angle_deg"]) % 180
            if diff > 90:
                diff = 180 - diff
            if (diff < angle_thresh
                    and math.hypot(obb["cx"] - k["cx"], obb["cy"] - k["cy"]) < dist_thresh):
                dup = True
                break
        if not dup:
            kept.append(obb)
    return kept


def hough_detect_streaks(img_path: pathlib.Path) -> list[dict]:
    """Run Hough line detection on one Frigate PNG.

    Frigate-tuned parameters (2325×1555, targeting 50–150 px short streaks):
        threshold=35   — was 80; a 50 px streak has ≤50 votes, so 80 is too high
        minLineLength=40 px — was 77 px (5% of 1555); that floor missed all <77 px streaks
        maxLineGap=10 px    — tighter than 25 px to avoid merging unrelated features
    """
    try:
        import cv2 as cv
    except ImportError:
        log.warning("opencv-python not available — skipping Hough detection")
        return []

    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape
    border = 40

    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(img)
    blurred = cv.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv.Canny(blurred, 30, 100)

    lines = cv.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=35,
        minLineLength=40,
        maxLineGap=10,
    )
    if lines is None:
        return []

    _MAX_WIDTH = 80.0
    obbs: list[dict] = []
    for line in lines:
        x1, y1, x2, y2 = map(float, line[0])
        cx_l = (x1 + x2) / 2
        cy_l = (y1 + y2) / 2

        if cx_l < border or cx_l > w - border or cy_l < border or cy_l > h - border:
            continue

        angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        width = estimate_streak_width(img, cx_l, cy_l, angle_deg)

        if width >= _MAX_WIDTH * 0.95:
            continue

        obb = endpoints_to_obb(x1, y1, x2, y2, width)
        obbs.append({k: round(v, 2) for k, v in obb.items()})

    obbs.sort(key=lambda o: -o["w"])
    return _deduplicate_obbs(obbs)[:5]


def precompute_all_suggestions(
    images: list[dict],
    suggestions_path: pathlib.Path,
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
        cache[str(entry["path"])] = hough_detect_streaks(entry["path"])
        if (i + 1) % 10 == 0:
            suggestions_path.parent.mkdir(parents=True, exist_ok=True)
            suggestions_path.write_text(json.dumps(cache, indent=2))

    suggestions_path.parent.mkdir(parents=True, exist_ok=True)
    suggestions_path.write_text(json.dumps(cache, indent=2))
    n_hit = sum(1 for v in cache.values() if v)
    log.info("Done. %d / %d images have ≥1 detection. Saved to %s",
             n_hit, len(cache), suggestions_path)
    return cache


# ---- data loading ------------------------------------------------------------

def _parse_filename(stem: str) -> tuple[int, str]:
    """Return (frame_number, timestamp_utc) from a Frigate PNG stem."""
    m = _FNAME_PAT.match(stem)
    if not m:
        return 0, ""
    ts = f"{m.group(2)}:{m.group(3)}:{m.group(4)}Z"
    return int(m.group(1)), ts


def _load_priority_list(json_path: pathlib.Path) -> dict[str, dict]:
    """Load screener JSON and return a mapping from filename to frame dict."""
    with open(json_path) as fh:
        data = json.load(fh)
    return {entry["frame"]: entry for entry in data.get("frames", [])}


def load_frigate_images(
    processed_dir: pathlib.Path,
    priority_map: dict[str, dict] | None = None,
    min_score: float = 0.0,
) -> list[dict]:
    """Build image list from a directory glob over processed PNGs.

    If priority_map is given, frames are sorted descending by score.
    Frames not in the map are appended at the end in chronological order.
    """
    all_pngs = sorted(processed_dir.glob("*.png"))
    if not all_pngs:
        raise FileNotFoundError(f"No PNG files found in {processed_dir}")

    entries: list[dict] = []
    for path in all_pngs:
        frame_num, ts = _parse_filename(path.stem)
        score = 0.0
        prev_frame = None
        if priority_map and path.name in priority_map:
            pentry = priority_map[path.name]
            score = pentry.get("score", 0.0)
            prev_name = pentry.get("prev_frame")
            if prev_name:
                prev_frame = processed_dir / prev_name
        entries.append({
            "path": path,
            "frame_number": frame_num,
            "timestamp_utc": ts,
            "score": score,
            "prev_path": prev_frame,
        })

    if priority_map:
        # Separate ranked (in priority list) and unranked frames
        ranked   = [e for e in entries if e["path"].name in priority_map
                    and e["score"] >= min_score]
        unranked = [e for e in entries if e["path"].name not in priority_map]
        ranked.sort(key=lambda e: -e["score"])
        entries = ranked + unranked

    return entries


def load_existing_annotations(output_path: pathlib.Path) -> dict[str, Any]:
    if output_path.exists():
        with open(output_path) as fh:
            return json.load(fh)
    return {
        "info": {
            "description": "Frigate satellite streak OBB annotations",
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
    ) -> None:
        super().__init__()
        self.title("ARGUS Frigate Streak Annotator")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)

        self.images = images
        self.coco = coco
        self.output_path = output_path
        self.suggestions = suggestions or {}
        self.idx = 0

        # zoom / pan state
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start: tuple[int, int] | None = None

        # per-image annotation state
        self._pending_a: tuple[float, float] | None = None
        self._selected_obb_idx: int | None = None
        self._img_obbs: list[dict] = []
        self._img_suggestions: list[dict] = []
        self._show_suggestions: bool = bool(suggestions)  # off by default if no cache

        # cached display image
        self._photo: ImageTk.PhotoImage | None = None
        self._photo_zoom: float | None = None
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
            canvas_frame,
            width=CANVAS_W,
            height=CANVAS_H,
            bg="#000010",
            cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        bot = tk.Frame(self, bg="#1a1a2e")
        bot.pack(fill="x", padx=6, pady=4)

        tk.Label(bot, text="Width (px):", fg="#aaaacc", bg="#1a1a2e").pack(side="left")
        self.width_var = tk.IntVar(value=12)
        self.width_var.trace_add("write", lambda *_: self._draw_overlays())
        tk.Scale(
            bot,
            from_=1, to=80,
            orient="horizontal",
            variable=self.width_var,
            bg="#1a1a2e", fg="#aaaacc",
            highlightthickness=0,
            troughcolor="#333355",
            length=160,
            takefocus=0,
        ).pack(side="left", padx=6)

        for txt, cmd in [
            ("◀  [A]",        self._prev),
            ("[D]  ▶",        self._next),
            ("Accept  [Y]",   self._accept_suggestion),
            ("Dismiss  [Esc]", self._cancel_pending),
            ("Delete  [Del]", self._delete_selected),
            ("Save  [S]",     self._save),
        ]:
            tk.Button(
                bot, text=txt, command=cmd,
                bg="#2a2a4a", fg="#ccccee", relief="flat", padx=7, pady=2,
            ).pack(side="left", padx=3)

        self.btn_hints = tk.Button(
            bot, text="Hints  [H]", command=self._toggle_suggestions,
            bg="#2a2a4a", fg="#ffaa33", relief="flat", padx=7, pady=2,
        )
        self.btn_hints.pack(side="left", padx=3)

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
        self.bind("<s>",         lambda _: self._save())
        self.bind("<q>",         lambda _: self._quit())
        self.bind("<h>",         lambda _: self._toggle_suggestions())

        # Mouse bindings apply to left panel only; right panel is read-only.
        self.canvas.bind("<Button-1>",        self._on_click)
        self.canvas.bind("<Button-2>",        self._on_pan_start)
        self.canvas.bind("<B2-Motion>",       self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.canvas.bind("<Button-3>",        self._on_pan_start)
        self.canvas.bind("<B3-Motion>",       self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>",      self._on_scroll)  # macOS
        self.canvas.bind("<Button-4>",        self._on_scroll)  # Linux up
        self.canvas.bind("<Button-5>",        self._on_scroll)  # Linux down

    # ---- coordinate helpers (left panel only) --------------------------------

    def _canvas_to_img(self, cx: float, cy: float) -> tuple[float, float]:
        """Convert left-panel canvas coordinates to image pixel coordinates."""
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
            pil = Image.open(entry["path"]).convert("RGB")
        except Exception as exc:
            log.error("Cannot open %s: %s", entry["path"], exc)
            pil = Image.new("RGB", (2325, 1555), (30, 30, 50))
        self._pil_img = pil

        self._photo = None
        self._photo_zoom = None

        # Fit to canvas
        self.zoom = min(CANVAS_W / pil.width, CANVAS_H / pil.height)
        self.pan_x = (CANVAS_W - pil.width  * self.zoom) / 2
        self.pan_y = (CANVAS_H - pil.height * self.zoom) / 2

        # Load confirmed OBBs
        img_id = self._get_or_create_image_id(entry)
        self._img_obbs = [
            ann["obb"]
            for ann in self.coco["annotations"]
            if ann["image_id"] == img_id
        ]

        # Load Hough suggestions
        key = str(entry["path"])
        self._img_suggestions = list(self.suggestions.get(key, []))

        self._pending_a = None
        self._selected_obb_idx = None

        if self._img_suggestions and not self._img_obbs:
            self.width_var.set(int(round(self._img_suggestions[0]["h"])))

        self._redraw()
        self._update_labels()

    def _get_or_create_image_id(self, entry: dict) -> int:
        fname = str(entry["path"])
        for img in self.coco["images"]:
            if img["file_name"] == fname:
                return img["id"]
        new_id = max((i["id"] for i in self.coco["images"]), default=0) + 1
        self.coco["images"].append({
            "id": new_id,
            "file_name": fname,
            "width": self._pil_img.width if self._pil_img else 2325,
            "height": self._pil_img.height if self._pil_img else 1555,
            "date_captured": entry.get("timestamp_utc", ""),
            "source": "frigate_QHY600M",
            "frame_number": entry.get("frame_number", 0),
            "screen_score": entry.get("score", 0.0),
        })
        return new_id

    # ---- drawing -------------------------------------------------------------

    def _redraw(self) -> None:
        """Full redraw: render only the visible crop of the image, then overlays."""
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
            self._photo = ImageTk.PhotoImage(resized)
            anchor_x = self.pan_x + vis_x0 * self.zoom
            anchor_y = self.pan_y + vis_y0 * self.zoom
            self.canvas.create_image(anchor_x, anchor_y, anchor="nw",
                                     image=self._photo, tags="bg")

        self._draw_overlays()

    def _toggle_suggestions(self) -> None:
        """Toggle Hough suggestion overlays on/off."""
        self._show_suggestions = not self._show_suggestions
        label = "Hints  [H]" if not self._show_suggestions else "Hints ✓ [H]"
        self.btn_hints.config(text=label)
        self._draw_overlays()
        self._update_labels()

    def _draw_overlays(self) -> None:
        """Redraw suggestion / OBB / pending overlays on the left panel."""
        self.canvas.delete("suggestion")
        self.canvas.delete("confirmed")
        self.canvas.delete("pending")

        # Suggestions — amber dashed polygons (only when enabled)
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
                cx, cy - 10,
                text=f"? {i + 1}  w={obb['h']:.0f}px",
                fill=SUGGESTION_COLOR,
                font=("Helvetica", 8),
                tags="suggestion",
            )

        # Confirmed OBBs — solid coloured polygons
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

        # Pending first click — dashed circle whose diameter = streak width
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
        done = sum(
            1 for img in self.coco["images"]
            if any(a["image_id"] == img["id"] for a in self.coco["annotations"])
        )
        entry = self.images[self.idx]
        self.lbl_progress.config(text=f"Image {self.idx + 1} / {n}  |  Annotated: {done}")
        self.lbl_fname.config(text=entry["path"].name[:60])
        self.lbl_count.config(text=f"OBBs: {len(self._img_obbs)}")
        self.lbl_zoom.config(text=f"zoom {self.zoom:.2f}×")

        score = entry.get("score", 0.0)
        if score > 0:
            self.lbl_score.config(text=f"screen score: {score:.2f}")
        else:
            self.lbl_score.config(text="")

        visible = self._img_suggestions if self._show_suggestions else []
        if self._pending_a is not None:
            hint, color = "→ Click streak END", "#88ffcc"
        elif visible:
            hint = f"Y = accept {len(visible)} suggestion(s)  |  or click to annotate manually"
            color = SUGGESTION_COLOR
        else:
            hint, color = "Click streak START", "#88ffcc"
        self.lbl_hint.config(text=hint, fg=color)

    # ---- interaction ---------------------------------------------------------

    def _on_click(self, event: tk.Event) -> None:
        ix, iy = self._canvas_to_img(event.x, event.y)
        img_w = self._pil_img.width  if self._pil_img else 2325
        img_h = self._pil_img.height if self._pil_img else 1555
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

    # ---- navigation ----------------------------------------------------------

    def _next(self) -> None:
        self._autosave()
        if self.idx < len(self.images) - 1:
            self._load_image(self.idx + 1)

    def _prev(self) -> None:
        self._autosave()
        if self.idx > 0:
            self._load_image(self.idx - 1)

    # ---- annotation actions --------------------------------------------------

    def _accept_suggestion(self) -> None:
        if not self._img_suggestions:
            return
        for obb in self._img_suggestions:
            self._img_obbs.append(obb)
        self.width_var.set(int(round(self._img_suggestions[0]["h"])))
        self._img_suggestions = []
        self._commit_obbs()
        self._draw_overlays()
        self._update_labels()

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
                "attributes": {"source": "frigate_manual"},
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
        "--processed-dir", type=pathlib.Path,
        default=pathlib.Path("/Volumes/External/frigate/processed"),
        help="Directory containing Frigate processed PNGs.",
    )
    parser.add_argument(
        "--raw-dir", type=pathlib.Path, default=None,
        help="Optional: Frigate raw FITS directory (for DATE-OBS metadata lookup).",
    )
    parser.add_argument(
        "--output", type=pathlib.Path,
        default=pathlib.Path("data/annotations/frigate_streaks.json"),
        help="Output COCO JSON (created fresh or resumed).",
    )
    parser.add_argument(
        "--priority-list", type=pathlib.Path, default=None,
        metavar="SCREEN_JSON",
        help="frigate_screen.json from screen_frigate.py — frames sorted by score.",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Skip frames below this screener score when --priority-list is given.",
    )
    parser.add_argument(
        "--no-suggestions", action="store_true",
        help="Ignore the Hough suggestion cache and annotate fully manually.",
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
        "--start-at", type=int, default=0,
        help="0-based image index to start at.",
    )
    args = parser.parse_args()

    suggestions_path = args.output.parent / (args.output.stem + ".suggestions.json")

    # Load screener priority list
    priority_map: dict[str, dict] | None = None
    if args.priority_list is not None:
        if not args.priority_list.exists():
            log.error("Priority list not found: %s", args.priority_list)
            sys.exit(1)
        priority_map = _load_priority_list(args.priority_list)
        log.info("Loaded priority list: %d entries", len(priority_map))

    log.info("Loading Frigate images from %s", args.processed_dir)
    try:
        images = load_frigate_images(args.processed_dir, priority_map, args.min_score)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    log.info("Loaded %d images", len(images))

    if args.precompute:
        precompute_all_suggestions(images, suggestions_path, force=args.force_recompute)
        return

    # Load suggestion cache if present (skip if --no-suggestions)
    suggestions: dict[str, list[dict]] = {}
    if args.no_suggestions:
        log.info("Suggestions disabled (--no-suggestions)")
    elif suggestions_path.exists():
        with open(suggestions_path) as fh:
            suggestions = json.load(fh)
        n_with = sum(1 for v in suggestions.values() if v)
        log.info("Loaded suggestions: %d images, %d with detections", len(suggestions), n_with)
    else:
        log.info("No suggestion cache — run with --precompute first for auto-hints")

    coco = load_existing_annotations(args.output)
    already_done = sum(
        1 for img in coco["images"]
        if any(a["image_id"] == img["id"] for a in coco["annotations"])
    )
    if already_done:
        log.info("Resuming — %d images already annotated", already_done)

    app = AnnotationApp(
        images, coco, args.output,
        suggestions=suggestions,
    )
    if args.start_at:
        app._load_image(args.start_at)

    app.mainloop()

    save_annotations(coco, args.output)
    n_ann = len(coco["annotations"])
    n_img = len([i for i in coco["images"]
                 if any(a["image_id"] == i["id"] for a in coco["annotations"])])
    log.info("Done. %d OBBs across %d images → %s", n_ann, n_img, args.output)


if __name__ == "__main__":
    main()
