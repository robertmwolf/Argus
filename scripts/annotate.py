"""Unified ARGUS streak OBB annotation tool.

Accepts FITS, PNG, or JPEG image directories. Outputs COCO JSON compatible
with merge_annotations.py.

Replaces the old dataset-specific streak labelers:
annotate_streaks.py, annotate_frigate_streaks.py, and
annotate_brentimages_streaks.py.

Usage:
    # Frigate processed PNGs:
    python scripts/annotate.py \\
        --image-dir /Volumes/External/TrainingData/raw/frigate/processed \\
        --output data/annotations/frigate_streaks.json

    # Any FITS directory:
    python scripts/annotate.py \\
        --image-dir /path/to/fits \\
        --output data/annotations/my_dataset.json

    # BrentImages FITS night (COCO JSON is the only write target):
    python scripts/annotate.py \\
        --night-dir /Volumes/External/TrainingData/raw/BrentImages/Img_20260515_Atwood

Keybindings:
    Escape              — cancel pending click
    Right / D / Space   — next image
    Left  / A           — previous image
    B                   — mark as blank (no streak), advance to next
    R                   — reject unusable, advance to next
    C                   — confirm SkyTrack annotation as-is, advance to next
    Delete / BackSpace  — delete selected OBB
    S                   — save now
    Q                   — quit and save
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import sys
import tkinter as tk
from datetime import datetime, timezone
from tkinter import ttk
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageTk

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---- visual constants --------------------------------------------------------
CANVAS_W = 1600
CANVAS_H = 820

COLORS = ["#00ff88", "#ff6644", "#44aaff", "#ffdd00", "#ff44ff"]
SEL_COLOR = "#ffffff"
BLANK_COLOR = "#ff4444"
UNREVIEWED_COLOR = "#ffaa00"
OBB_LINE_WIDTH = 2
DEFAULT_WIDTH = 10
DEFAULT_STRK_WIDTH = 16.0
UNUSABLE_REJECT_CODE = "5"
UNUSABLE_REJECT_COMMENT = "Rejected unusable - do not use"

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



# ---- image list loading ------------------------------------------------------
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


# ---- BrentImages .strk support ----------------------------------------------

def load_strk_lines(strk_path: pathlib.Path) -> list[str]:
    """Read a BrentImages .strk file as raw lines."""
    return strk_path.read_text(encoding="utf-8").splitlines()


def write_strk_lines(strk_path: pathlib.Path, lines: list[str]) -> None:
    """Atomically write updated BrentImages .strk lines."""
    tmp = strk_path.with_suffix(".strk.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(strk_path)


def update_strk_obs(
    lines: list[str],
    filename: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    reject: str,
    jd: float = 0.0,
    exposure: float = 0.5,
    gain: float = 0.0,
) -> list[str]:
    """Update one OBS row in a BrentImages .strk file.

    Args:
        lines: Existing .strk file lines.
        filename: FITS basename to update.
        x1: Streak start x pixel.
        y1: Streak start y pixel.
        x2: Streak end x pixel.
        y2: Streak end y pixel.
        reject: "0" for streak, "-1" for no streak, "2" for pending.
        jd: Julian date midpoint.
        exposure: Exposure time in seconds.
        gain: Camera gain.

    Returns:
        Updated .strk file lines.
    """
    dx, dy = x2 - x1, y2 - y1
    length = round(math.hypot(dx, dy), 1)
    cx = round((x1 + x2) / 2, 1)
    cy = round((y1 + y2) / 2, 1)
    elongation = round(math.degrees(math.atan2(abs(dy), abs(dx))), 1) if length > 0 else 0.0

    new_fields = [
        filename,
        "",
        f"{jd:.10f}",
        f"{x1:.0f}", f"{y1:.0f}",
        f"{x2:.0f}", f"{y2:.0f}",
        f"{cx:.1f}", f"{cy:.1f}",
        "0", "0",
        f"{elongation:.1f}",
        f"{length:.1f}",
        reject,
        "0", "0",
        "0", "0",
        "0", "0",
        "0", "0",
        "0",
        f"{exposure:.4f}",
        f"{gain:.1f}",
        "Manual annotation" if reject == "0" else (
            "Confirmed no streak" if reject == "-1" else (
                UNUSABLE_REJECT_COMMENT
                if reject == UNUSABLE_REJECT_CODE
                else "Pending manual annotation"
            )
        ),
    ]

    result: list[str] = []
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
                orig_date = parts[1].strip() if len(parts) > 1 else ""
                if orig_date:
                    new_fields[1] = orig_date
                    if jd == 0.0:
                        try:
                            from astropy.time import Time
                            new_fields[2] = f"{Time(orig_date, format='isot', scale='utc').jd:.10f}"
                        except Exception:
                            pass
                result.append("\t".join(new_fields))
                continue
        result.append(line)
    return result


def load_night_frames(night_dir: pathlib.Path) -> list[dict]:
    """Load BrentImages FITS frames from a night directory of .strk stubs."""
    strk_files = sorted(night_dir.glob("*.strk"))
    if not strk_files:
        raise FileNotFoundError(
            f"No .strk files found in {night_dir} - run generate_brentimages_strk.py first"
        )

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
                in_obs = False
                continue
            if stripped.startswith("Image\t"):
                in_obs = True
                continue
            if stripped.startswith("[") or not stripped:
                in_obs = False
                continue

            parts = stripped.split("\t")
            if (
                not in_obs
                and len(parts) >= 16
                and parts[0].strip().isdigit()
                and not parts[0].strip().startswith("Image")
            ):
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
                        "path": fits_path,
                        "fits_path": fits_path,
                        "filename": fname,
                        "norad_id": norad_id,
                        "sat_name": sat_name,
                        "strk_path": strk_path,
                        "date_obs": parts[1].strip(),
                        "jd": float(parts[2]) if parts[2].strip() else 0.0,
                        "x_start": float(parts[3]) if parts[3].strip() else 0.0,
                        "y_start": float(parts[4]) if parts[4].strip() else 0.0,
                        "x_end": float(parts[5]) if parts[5].strip() else 0.0,
                        "y_end": float(parts[6]) if parts[6].strip() else 0.0,
                        "reject": parts[13].strip(),
                        "comment": parts[-1].strip() if len(parts) > 25 else "",
                        "exposure": float(parts[23].strip()) if len(parts) > 23 and parts[23].strip() else 0.5,
                        "gain": float(parts[24].strip()) if len(parts) > 24 and parts[24].strip() else 0.0,
                    })
                except (ValueError, IndexError):
                    continue

        if obs_entries and norad_id is not None:
            obs_entries.sort(key=lambda e: e["date_obs"])
            passes.append(obs_entries)

    frames: list[dict] = []
    for pass_idx, pass_frames in enumerate(passes):
        for frame_idx, frame in enumerate(pass_frames):
            frame["pass_idx"] = pass_idx
            frame["frame_in_pass"] = frame_idx
            frame["total_in_pass"] = len(pass_frames)
            frame["num_passes"] = len(passes)
            frame["score"] = 0.0
            frames.append(frame)
    return frames


def seed_coco_from_strk(coco: dict[str, Any], frames: list[dict], source_name: str) -> None:
    """Populate COCO state from existing BrentImages .strk decisions.

    All decisions seeded from .strk files are marked needs_review=True because
    SkyTrack coordinates are auto-detected and have not been human-verified.
    The annotator's C key (confirm) or any manual edit clears this flag.

    Also retroactively flags images already present in the COCO JSON that were
    seeded in a previous session (recognisable because their annotations still
    carry strk_path in attributes, meaning no human ever replaced them).
    """
    # --- retroactively flag previously-seeded images that were never confirmed ---
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img in coco.get("images", []):
        if "needs_review" in img:
            continue  # already has an explicit flag — respect it
        if img.get("rejected"):
            continue  # human-rejected; leave alone
        if not img.get("strk_path"):
            continue  # not from a .strk seed; skip

        img_anns = anns_by_image.get(img["id"], [])
        # Human-drawn OBBs never carry strk_path in attributes.
        # If every annotation still has strk_path, no human has touched this image.
        has_human_ann = any(
            not ann.get("attributes", {}).get("strk_path") for ann in img_anns
        )
        if has_human_ann:
            continue  # human already reviewed and replaced annotations

        # SkyTrack-seeded blank or annotation that was never confirmed.
        img["needs_review"] = True
        for ann in img_anns:
            ann.setdefault("attributes", {})["needs_review"] = True

    # --- seed new frames not yet in the COCO JSON ---
    existing = {img["file_name"] for img in coco.get("images", [])}
    max_img_id = max((img["id"] for img in coco.get("images", [])), default=0)
    max_ann_id = max((ann["id"] for ann in coco.get("annotations", [])), default=0)

    for frame in frames:
        fname = str(frame["path"])
        if fname in existing:
            continue

        is_unusable = (
            frame.get("reject") not in ("0", "-1", "2")
            or frame.get("comment") == UNUSABLE_REJECT_COMMENT
        )
        # Only needs_review for non-unusable frames — unusable ones are just skipped.
        needs_review = not is_unusable

        max_img_id += 1
        coco["images"].append({
            "id": max_img_id,
            "file_name": fname,
            "width": 0,
            "height": 0,
            "date_captured": frame.get("date_obs", ""),
            "source": source_name,
            "norad_id": frame.get("norad_id"),
            "sat_name": frame.get("sat_name", ""),
            "strk_path": str(frame.get("strk_path", "")),
            "frame_in_pass": frame.get("frame_in_pass", 0),
            "pass_idx": frame.get("pass_idx", 0),
            "blank": frame.get("reject") == "-1",
            "rejected": is_unusable,
            "needs_review": needs_review,
            "exclude_reason": "rejected_unusable" if is_unusable else "",
        })
        existing.add(fname)

        if frame.get("reject") == "0" and frame.get("x_end", 0.0) != 0.0:
            obb = endpoints_to_obb(
                frame["x_start"],
                frame["y_start"],
                frame["x_end"],
                frame["y_end"],
                DEFAULT_STRK_WIDTH,
            )
            max_ann_id += 1
            coco["annotations"].append({
                "id": max_ann_id,
                "image_id": max_img_id,
                "category_id": 1,
                "bbox": obb_to_bbox(obb),
                "area": round(obb["w"] * obb["h"], 1),
                "iscrowd": 0,
                "segmentation": [[coord for corner in obb_corners(**obb) for coord in corner]],
                "obb": {k: round(v, 2) for k, v in obb.items()},
                "attributes": {
                    "source": source_name,
                    "strk_path": str(frame["strk_path"]),
                    "needs_review": True,
                },
            })


# ---- main application --------------------------------------------------------

class AnnotationApp(tk.Tk):
    def __init__(
        self,
        images: list[dict],
        coco: dict[str, Any],
        output_path: pathlib.Path,
        source_name: str = "manual",
        demo_dir: pathlib.Path | None = None,
        save_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__()
        self.title("ARGUS Streak Annotator")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w = min(CANVAS_W + 40, sw - 80)
        h = min(CANVAS_H + 140, sh - 80)
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        self.all_images = images
        self.images = images
        self.coco = coco
        self.output_path = output_path
        self.source_name = source_name
        self.demo_dir = demo_dir or pathlib.Path.home() / "Desktop" / "Demo Images"
        self.save_hook = save_hook
        self.idx = 0
        self._pending_only: bool = False

        # zoom / pan
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start: tuple[int, int] | None = None

        # per-image state
        self._pending_a: tuple[float, float] | None = None
        self._selected_obb_idx: int | None = None
        self._img_obbs: list[dict] = []
        # parallel list: True = OBB came from SkyTrack and hasn't been confirmed
        self._img_obbs_needs_review: list[bool] = []

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
        goto_entry.bind("<Return>", self._on_goto_return)
        tk.Button(
            top, text="Go", command=self._goto_index,
            bg="#2a2a4a", fg="#ccccee", relief="flat", padx=7, pady=1,
        ).pack(side="left", padx=(3, 0))

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
            ("Blank  [B]",    self._mark_blank),
            ("Reject  [R]",   self._mark_rejected),
            ("Confirm  [C]",  self._confirm_review),
            ("Cancel  [Esc]", self._cancel_pending),
            ("Delete  [Del]", self._delete_selected),
            ("Save  [S]",     self._save),
        ]:
            tk.Button(
                bot, text=txt, command=cmd,
                bg="#2a2a4a", fg="#ccccee", relief="flat", padx=7, pady=2,
            ).pack(side="left", padx=3)

        self.btn_pending = tk.Button(
            bot, text="Pending only  [P]", command=self._toggle_pending_only,
            bg="#2a2a4a", fg="#ccccee", relief="flat", padx=7, pady=2,
        )
        self.btn_pending.pack(side="left", padx=3)

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
        self.bind("<b>",         lambda _: self._mark_blank())
        self.bind("<r>",         lambda _: self._mark_rejected())
        self.bind("<c>",         lambda _: self._confirm_review())
        self.bind("<s>",         lambda _: self._save())
        self.bind("<q>",         lambda _: self._quit())
        self.bind("<p>",         lambda _: self._toggle_pending_only())
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

    def _image_needs_review(self, img_id: int) -> bool:
        for img in self.coco["images"]:
            if img["id"] == img_id:
                return bool(img.get("needs_review"))
        return False

    def _reviewed_file_names(self) -> set[str]:
        """Return image file names that have a final human-confirmed decision."""
        annotated_ids = {a["image_id"] for a in self.coco["annotations"]}
        reviewed: set[str] = set()
        for img in self.coco["images"]:
            if img.get("needs_review"):
                # SkyTrack-seeded data pending human confirmation — not reviewed yet.
                continue
            if img.get("rejected") or img.get("blank") or img["id"] in annotated_ids:
                reviewed.add(img["file_name"])
        return reviewed

    def _is_file_reviewed(self, file_name: str | None) -> bool:
        """Return whether one image file already has a final review decision."""
        return bool(file_name) and file_name in self._reviewed_file_names()

    def _current_file_name(self) -> str | None:
        """Return the current image file name, if the queue is non-empty."""
        if not self.images:
            return None
        return str(self.images[self.idx]["path"])

    def _set_queue_for_filter(self, preferred_file: str | None = None) -> None:
        """Rebuild the visible queue after changing the pending-only filter."""
        if self._pending_only:
            reviewed = self._reviewed_file_names()
            self.images = [
                entry for entry in self.all_images if str(entry["path"]) not in reviewed
            ]
        else:
            self.images = self.all_images

        if not self.images:
            self.idx = 0
            self._pil_img = Image.new("RGB", (1200, 800), (30, 30, 50))
            self._photo = None
            self._img_obbs = []
            self._img_obbs_needs_review = []
            self._pending_a = None
            self._selected_obb_idx = None
            self.canvas.delete("all")
            self.canvas.create_text(
                CANVAS_W // 2,
                CANVAS_H // 2,
                text="No pending images",
                fill="#88ffcc",
                font=("Helvetica", 18, "bold"),
            )
            self._update_labels()
            return

        if preferred_file:
            for i, entry in enumerate(self.images):
                if str(entry["path"]) == preferred_file:
                    self._load_image(i)
                    return

        if preferred_file:
            all_index = next(
                (
                    i for i, entry in enumerate(self.all_images)
                    if str(entry["path"]) == preferred_file
                ),
                -1,
            )
            if all_index >= 0:
                for i, entry in enumerate(self.images):
                    try:
                        if self.all_images.index(entry) > all_index:
                            self._load_image(i)
                            return
                    except ValueError:
                        continue

        self._load_image(min(self.idx, len(self.images) - 1))

    def _load_image(self, idx: int) -> None:
        if not self.images:
            return
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
        anns = [a for a in self.coco["annotations"] if a["image_id"] == img_id]
        self._img_obbs = [ann["obb"] for ann in anns]
        self._img_obbs_needs_review = [
            bool(ann.get("attributes", {}).get("needs_review")) for ann in anns
        ]

        self._pending_a = None
        self._selected_obb_idx = None

        self._redraw()
        self._update_labels()

    def _is_blank(self, img_id: int) -> bool:
        for img in self.coco["images"]:
            if img["id"] == img_id:
                return img.get("blank", False)
        return False

    def _is_rejected(self, img_id: int) -> bool:
        for img in self.coco["images"]:
            if img["id"] == img_id:
                return img.get("rejected", False)
        return False

    def _get_or_create_image_id(self, entry: dict) -> int:
        fname = str(entry["path"])
        for img in self.coco["images"]:
            if img["file_name"] == fname:
                if self._pil_img is not None and (not img.get("width") or not img.get("height")):
                    img["width"] = self._pil_img.width
                    img["height"] = self._pil_img.height
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
            if self._is_rejected(img_id):
                r, g, b = resized.split()
                r = r.point(lambda v: int(v * 0.55))
                g = g.point(lambda v: int(v * 0.55))
                b = b.point(lambda v: min(255, int(v * 1.25)))
                resized = Image.merge("RGB", (r, g, b))
            elif self._is_blank(img_id):
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
        self.canvas.delete("confirmed")
        self.canvas.delete("pending")
        self.canvas.delete("blank_label")
        self.canvas.delete("reject_label")
        self.canvas.delete("review_banner")

        img_id = self._get_or_create_image_id(self.images[self.idx])
        cw = self.canvas.winfo_width() or CANVAS_W

        if self._is_rejected(img_id):
            self.canvas.create_text(
                cw // 2, 30, text="REJECTED UNUSABLE (not training/eval)",
                fill="#88aaff", font=("Helvetica", 16, "bold"), tags="reject_label",
            )
            return
        if self._is_blank(img_id):
            self.canvas.create_text(
                cw // 2, 30, text="✗  CONFIRMED BLANK (no streak)",
                fill=BLANK_COLOR, font=("Helvetica", 16, "bold"), tags="blank_label",
            )
            return

        # Show review banner when any OBB (or the image itself) needs review.
        if self._image_needs_review(img_id):
            self.canvas.create_text(
                cw // 2, 22,
                text="⚠  SKYTRACK AUTO-DETECTION — verify OBBs then C to confirm  |  B = blank  |  Del = redraw",
                fill=UNREVIEWED_COLOR, font=("Helvetica", 13, "bold"), tags="review_banner",
            )

        for i, obb in enumerate(self._img_obbs):
            unreviewed = (
                i < len(self._img_obbs_needs_review) and self._img_obbs_needs_review[i]
            )
            if i == self._selected_obb_idx:
                color = SEL_COLOR
            elif unreviewed:
                color = UNREVIEWED_COLOR
            else:
                color = COLORS[i % len(COLORS)]

            corners = obb_corners(obb["cx"], obb["cy"], obb["w"], obb["h"], obb["angle_deg"])
            pts = self._corners_to_canvas(corners)
            self.canvas.create_polygon(
                pts, outline=color, fill="", width=OBB_LINE_WIDTH, tags="confirmed"
            )
            cx, cy = self._img_to_canvas(obb["cx"], obb["cy"])
            r = 4
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    outline=color, fill=color, tags="confirmed")
            label = f"#{i + 1}" + (" ?" if unreviewed else "")
            self.canvas.create_text(cx + 8, cy - 8, text=label,
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
        if not self.images:
            reviewed = len(self._reviewed_file_names())
            total = len(self.all_images)
            self.lbl_progress.config(text=f"No pending images  |  Reviewed: {reviewed} / {total}")
            self.lbl_fname.config(text="")
            self.lbl_score.config(text="")
            self.lbl_count.config(text="DONE", fg="#66ff88")
            self.lbl_zoom.config(text="")
            self.btn_pending.config(text="Pending ✓ [P]", fg="#88ffcc")
            self.lbl_hint.config(text="All visible images are reviewed", fg="#88ffcc")
            return

        n = len(self.images)
        annotated_ids = {a["image_id"] for a in self.coco["annotations"]}
        blank_ids = {img["id"] for img in self.coco["images"] if img.get("blank")}
        rejected_ids = {img["id"] for img in self.coco["images"] if img.get("rejected")}
        image_id_by_file = {img["file_name"]: img["id"] for img in self.coco["images"]}
        queue_ids = {
            image_id_by_file[str(entry["path"])]
            for entry in self.images
            if str(entry["path"]) in image_id_by_file
        }
        done = len((annotated_ids | blank_ids | rejected_ids) & queue_ids)
        total_done = len(self._reviewed_file_names())
        filter_text = "pending" if self._pending_only else "all"
        self.lbl_progress.config(
            text=(
                f"Image {self.idx + 1} / {n} ({filter_text})  |  "
                f"Done: {done} visible, {total_done} / {len(self.all_images)} total"
            )
        )
        self.lbl_fname.config(text=self.images[self.idx]["path"].name[:70])

        score = self.images[self.idx].get("score", 0.0)
        self.lbl_score.config(text=f"score: {score:.2f}" if score else "")

        img_id = self._get_or_create_image_id(self.images[self.idx])
        cur_blank = self._is_blank(img_id)
        cur_rejected = self._is_rejected(img_id)
        cur_needs_review = self._image_needs_review(img_id)

        count_text = (
            "REJECT"
            if cur_rejected
            else ("BLANK" if cur_blank else f"OBBs: {len(self._img_obbs)}")
        )
        count_color = "#88aaff" if cur_rejected else (BLANK_COLOR if cur_blank else "#66ff88")
        self.lbl_count.config(text=count_text, fg=count_color)
        self.lbl_zoom.config(text=f"zoom {self.zoom:.2f}×")

        pending_label = "Pending ✓ [P]" if self._pending_only else "Pending only  [P]"
        pending_color = "#88ffcc" if self._pending_only else "#ccccee"
        self.btn_pending.config(text=pending_label, fg=pending_color)

        if cur_rejected:
            hint, color = "REJECTED — press R to undo", "#88aaff"
        elif cur_blank and cur_needs_review:
            hint = "SkyTrack: blank — C to confirm, or draw streak if wrong"
            color = UNREVIEWED_COLOR
        elif cur_blank:
            hint, color = "BLANK — press B to undo", BLANK_COLOR
        elif cur_needs_review:
            hint = "SkyTrack OBB (orange) — C to confirm  |  Del+redraw to fix  |  B if blank"
            color = UNREVIEWED_COLOR
        elif self._pending_a is not None:
            hint, color = "→ Click streak END", "#88ffcc"
        else:
            hint = "Click streak START  |  B = blank  |  R = reject unusable"
            color = "#88ffcc"
        self.lbl_hint.config(text=hint, fg=color)

    # ---- navigation ----------------------------------------------------------

    def _next(self) -> None:
        current_file = self._current_file_name()
        self._autosave()
        if self._pending_only and self._is_file_reviewed(current_file):
            self._set_queue_for_filter(current_file)
            return
        if self.idx < len(self.images) - 1:
            self._load_image(self.idx + 1)

    def _prev(self) -> None:
        current_file = self._current_file_name()
        self._autosave()
        if self._pending_only and self._is_file_reviewed(current_file):
            self._set_queue_for_filter(current_file)
            return
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

    def _on_goto_return(self, _: tk.Event) -> str:
        """Handle Return in the Go-to field."""
        self._goto_index()
        self.focus_set()
        return "break"

    # ---- annotation actions --------------------------------------------------

    def _clear_needs_review(self, img_id: int) -> None:
        """Clear the needs_review flag from an image and all its annotations."""
        for img in self.coco["images"]:
            if img["id"] == img_id:
                img.pop("needs_review", None)
                break
        for ann in self.coco["annotations"]:
            if ann["image_id"] == img_id:
                ann.get("attributes", {}).pop("needs_review", None)
        self._img_obbs_needs_review = [False] * len(self._img_obbs)

    def _confirm_review(self) -> None:
        """Accept the current SkyTrack OBBs as-is, mark image reviewed, advance."""
        if not self.images:
            return
        entry = self.images[self.idx]
        img_id = self._get_or_create_image_id(entry)
        self._clear_needs_review(img_id)
        self._autosave()
        self._redraw()
        self._update_labels()
        self._next()

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
        self._img_obbs_needs_review = []

        for img in self.coco["images"]:
            if img["id"] == img_id:
                img["blank"] = not currently_blank
                img["rejected"] = False
                img.pop("exclude_reason", None)
                img.pop("needs_review", None)
                break

        self._autosave()
        if not currently_blank:
            self._next()
        else:
            self._redraw()
            self._update_labels()

    def _mark_rejected(self) -> None:
        """Toggle unusable reject status for the current image and advance."""
        entry = self.images[self.idx]
        img_id = self._get_or_create_image_id(entry)
        currently_rejected = self._is_rejected(img_id)

        self.coco["annotations"] = [
            a for a in self.coco["annotations"] if a["image_id"] != img_id
        ]
        self._img_obbs = []
        self._img_obbs_needs_review = []

        for img in self.coco["images"]:
            if img["id"] == img_id:
                img["rejected"] = not currently_rejected
                img["blank"] = False
                img.pop("needs_review", None)
                if currently_rejected:
                    img.pop("exclude_reason", None)
                else:
                    img["exclude_reason"] = "rejected_unusable"
                break

        self._autosave()
        if not currently_rejected:
            self._next()
        else:
            self._redraw()
            self._update_labels()

    def _toggle_pending_only(self) -> None:
        """Toggle hiding images that already have a review decision."""
        current_file = self._current_file_name()
        self._autosave()
        self._pending_only = not self._pending_only
        self._set_queue_for_filter(current_file)

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
        for img in self.coco["images"]:
            if img["id"] == img_id:
                img["blank"] = False
                img["rejected"] = False
                img.pop("exclude_reason", None)
                img.pop("needs_review", None)
                break
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
        # Any edit clears the needs_review state — the human has engaged with this image.
        self._img_obbs_needs_review = [False] * len(self._img_obbs)

    def _delete_selected(self) -> None:
        if self._selected_obb_idx is None or not self._img_obbs:
            return
        self._img_obbs.pop(self._selected_obb_idx)
        if self._selected_obb_idx < len(self._img_obbs_needs_review):
            self._img_obbs_needs_review.pop(self._selected_obb_idx)
        self._selected_obb_idx = None
        self._commit_obbs()
        self._draw_overlays()
        self._update_labels()

    def _cancel_pending(self) -> None:
        self._pending_a = None
        self._selected_obb_idx = None
        self._draw_overlays()
        self._update_labels()

    def _on_click(self, event: tk.Event) -> None:
        img_id = self._get_or_create_image_id(self.images[self.idx])
        if self._is_blank(img_id) or self._is_rejected(img_id):
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
                self._img_obbs_needs_review.append(False)
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
        if self.save_hook is not None:
            self.save_hook(self.coco)

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
        "--image-dir", type=pathlib.Path, default=None,
        help="Directory containing images (FITS, PNG, or JPEG).",
    )
    parser.add_argument(
        "--night-dir", type=pathlib.Path, default=None,
        help="BrentImages night directory containing FITS files and .strk stubs. "
             "Annotations are written to brentimages_annotations.json in that directory; "
             ".strk files are read for metadata only and are not modified.",
    )
    parser.add_argument(
        "--output", type=pathlib.Path,
        default=pathlib.Path("data/annotations/streak_annotations.json"),
        help="Output COCO JSON path (created fresh or resumed).",
    )
    parser.add_argument(
        "--source-name", default="",
        help="Source label written into annotation attributes.",
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
        "--review-reject", default=None,
        help=(
            "BrentImages only: comma-separated .strk Reject codes to review "
            "(for example, -1 for confirmed no-streak frames)."
        ),
    )
    parser.add_argument(
        "--demo-dir", type=pathlib.Path,
        default=pathlib.Path.home() / "Desktop" / "Demo Images",
        help="Folder where flagged demo images are copied (default: ~/Desktop/Demo Images).",
    )
    args = parser.parse_args()

    if bool(args.image_dir) == bool(args.night_dir):
        parser.error("Provide exactly one of --image-dir or --night-dir")

    strk_frames: list[dict] = []

    if args.night_dir:
        if args.output == pathlib.Path("data/annotations/streak_annotations.json"):
            args.output = args.night_dir / "brentimages_annotations.json"
        log.info("Loading BrentImages night from %s", args.night_dir)
        has_strk = any(args.night_dir.glob("*.strk"))
        if has_strk:
            try:
                strk_frames = load_night_frames(args.night_dir)
            except FileNotFoundError as exc:
                log.error("%s", exc)
                sys.exit(1)
            images = strk_frames
            log.info(
                "Loaded %d frames across %d passes",
                len(images),
                max((f["pass_idx"] for f in images), default=-1) + 1,
            )
            if args.review_reject:
                review_codes = {code.strip() for code in args.review_reject.split(",") if code.strip()}
                images = [frame for frame in images if str(frame.get("reject", "")).strip() in review_codes]
                log.info(
                    "Review filter Reject in %s: %d frames",
                    ",".join(sorted(review_codes)),
                    len(images),
                )
        else:
            log.info("No .strk files found — loading FITS directly from directory")
            try:
                images = load_from_dir(args.night_dir)
                log.info("Loaded %d images", len(images))
            except FileNotFoundError as exc:
                log.error("%s", exc)
                sys.exit(1)
    else:
        log.info("Loading image list from %s", args.image_dir)
        try:
            images = load_from_dir(args.image_dir, args.priority_list, args.min_score)
            log.info("Loaded %d images from directory", len(images))
        except FileNotFoundError as exc:
            log.error("%s", exc)
            sys.exit(1)

    if not images:
        log.error("No images found")
        sys.exit(1)

    source_name = args.source_name or (
        args.night_dir.name if args.night_dir else args.image_dir.name
    )
    coco = load_existing_annotations(args.output, source_name)
    if strk_frames:
        seed_coco_from_strk(coco, strk_frames, source_name)

    annotated_ids = {a["image_id"] for a in coco["annotations"]}
    blank_ids = {img["id"] for img in coco["images"] if img.get("blank")}
    rejected_ids = {img["id"] for img in coco["images"] if img.get("rejected")}
    needs_review_ids = {img["id"] for img in coco["images"] if img.get("needs_review")}
    already_done = len(annotated_ids | blank_ids | rejected_ids) - len(needs_review_ids)
    if already_done > 0:
        log.info(
            "Resuming — %d images human-reviewed, %d SkyTrack-seeded pending review",
            already_done, len(needs_review_ids),
        )

    # Determine start index (--start-at is 1-based; 0 means auto)
    if args.start_at > 0:
        start_idx = args.start_at - 1
    else:
        # Jump to first image that needs human attention
        img_needs_review = {img["file_name"] for img in coco["images"] if img.get("needs_review")}
        done_fnames = {
            img["file_name"] for img in coco["images"]
            if not img.get("needs_review")
            and (img["id"] in annotated_ids or img.get("blank") or img.get("rejected"))
        }
        start_idx = 0
        for i, entry in enumerate(images):
            fname = str(entry["path"])
            if fname in img_needs_review or fname not in done_fnames:
                start_idx = i
                break

    app = AnnotationApp(images, coco, args.output,
                        source_name=source_name,
                        demo_dir=args.demo_dir)
    app._load_image(start_idx)
    app.mainloop()

    save_annotations(coco, args.output)
    n_ann = len(coco["annotations"])
    n_img = len({a["image_id"] for a in coco["annotations"]})
    n_blank = sum(1 for img in coco["images"] if img.get("blank"))
    n_rejected = sum(1 for img in coco["images"] if img.get("rejected"))
    n_pending = sum(1 for img in coco["images"] if img.get("needs_review"))
    log.info(
        "Done. %d OBBs across %d images, %d blanks, %d rejected, %d pending review → %s",
        n_ann, n_img, n_blank, n_rejected, n_pending, args.output,
    )


if __name__ == "__main__":
    main()
