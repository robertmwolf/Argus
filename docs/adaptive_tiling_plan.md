# Adaptive Tiling — Design Plan

> **Status:** Planning. Not yet implemented. No code changes should be made
> until this document is reviewed and the open questions in §4 are resolved.

---

## 1. Problem Statement

The current tiling implementation in `inference/tiled_pipeline.py` uses a
single fixed parameter: `tile_size=400`. This means native pixels are cropped
at 400×400 and passed directly to the model at 400px input — a 1:1
magnification ratio. This works well when source image streaks are already
near the model's training sweet spot (~50–300px at model input), but fails at
both extremes:

| Dataset | Native res | Streak (native) | At model input — full resize | At model input — 1:1 tile | Result |
|---------|-----------|-----------------|------------------------------|--------------------------|--------|
| SatStreaks | ~3000×2000 | ~200–800px | ~26–107px | n/a (no tile) | ✅ trained distribution |
| BrentImages Night 2 | 6248×4176 | ~440–1450px | ~28–93px | 440–1450px | ✅ tiling fixes it |
| Frigate | 2325×1555 | **20–80px** | 3–14px | 20–80px | ❌ still too small |

Root cause: `native_tile_size` and `model_input_size` are currently fused
into a single `tile_size` parameter, so magnification is always 1×.

---

## 2. Proposed Solution

### 2.1 Decouple Native Tile Size from Model Input Size

Replace the single `tile_size` parameter with two independent values:

```python
native_tile_size: int   # crop footprint in source image pixels
model_input_size: int   # resize target before model inference (= 400, always)
magnification = model_input_size / native_tile_size
```

The effect on streak appearance at model input:

```
streak_at_model_input = streak_native_px × magnification
                      = streak_native_px × (model_input_size / native_tile_size)
```

For Frigate (streaks ~40px native, target ~150px at model input):
```
native_tile_size = 400 × (40 / 150) ≈ 107px  →  use 110px
```

### 2.2 Automatic Parameter Selection

Add a helper `select_tile_params()` that chooses `native_tile_size` and
`overlap` from image metadata and estimated streak scale:

```python
def select_tile_params(
    image_shape: tuple[int, int],           # (H, W) in native pixels
    expected_streak_native_px: float,       # estimated streak length in source
    target_streak_model_px: float = 150,    # desired streak size at model input
    model_input_size: int = 400,
    min_native_tile: int = 80,              # floor: prevents absurdly small tiles
    max_downscale: float = 3.0,             # ceiling: no benefit beyond 3× downscale
) -> dict:
    """
    Returns {'native_tile_size': int, 'overlap': float}.
    
    Overlap is set so that tiles shift by at most max_streak_length pixels,
    preventing a streak from being entirely missed between tile boundaries.
    """
    native_tile_size = int(model_input_size * expected_streak_native_px / target_streak_model_px)
    native_tile_size = max(native_tile_size, min_native_tile)
    native_tile_size = min(native_tile_size, int(model_input_size * max_downscale))

    # overlap must cover at least one max streak
    max_streak_native = expected_streak_native_px * 3   # conservative upper bound
    min_overlap = max_streak_native / native_tile_size
    overlap = min(0.5, max(0.25, min_overlap))

    return dict(native_tile_size=native_tile_size, overlap=overlap)
```

### 2.3 Updated `run_tiled_inference` Signature

```python
def run_tiled_inference(
    image_array,
    model,
    native_tile_size: int = 400,    # was: tile_size
    model_input_size: int = 400,    # new: always 400 for current model
    overlap: float = 0.5,
    confidence_threshold: float = 0.05,
    nms_iou_threshold: float = 0.4,
    image_size: int | None = None,  # deprecated alias for model_input_size
)
```

`tile_image()` keeps its current interface but gains a `resize_to` parameter
so each crop is resized before returning:

```python
def tile_image(
    array,
    tile_size: int,          # native crop size
    overlap: float,
    resize_to: int | None,   # if set, resize each crop to this before yielding
) -> Iterator[tuple[np.ndarray, int, int]]:
```

### 2.4 Prediction Coordinate Remapping

`remap_predictions()` currently adds `(x0, y0)` to bbox coordinates assuming
1:1 native pixel scale. When `magnification != 1`, predictions come back in
*resized tile* coordinates and must be scaled back to native pixels before
remapping:

```python
def remap_predictions(preds, x0, y0, magnification=1.0):
    """
    1. Scale bbox by 1/magnification (tile coords → native tile coords)
    2. Translate by (x0, y0) (native tile coords → full image coords)
    """
    for p in preds:
        x, y, w, h = p["bbox"]
        yield {**p, "bbox": [
            x / magnification + x0,
            y / magnification + y0,
            w / magnification,
            h / magnification,
        ]}
```

---

## 3. Per-Dataset Recommended Parameters

| Dataset | Native res | Streak (native) | `native_tile_size` | magnification | overlap |
|---------|-----------|-----------------|-------------------|---------------|---------|
| SatStreaks | ~3000×2000 | 200–800px | 400 (or full image) | 1× | 0.5 |
| BrentImages Night 2 | 6248×4176 | 440–1450px | 400 | 1× | 0.5 |
| BrentImages Night 1 | 6248×4176 | (same) | 400 | 1× | 0.5 |
| Frigate | 2325×1555 | **20–80px** | **110** | **3.6×** | 0.5 |
| Unknown / auto | any | from metadata | `select_tile_params()` | adaptive | adaptive |

For Frigate, the expected improvement: streaks go from 20–80px at model input
(with 1:1 tiling) to ~70–290px — well within the training sweet spot.

---

## 4. Open Questions

1. **Frigate streaks may be sub-patch after all** — ViT-B/16 patches are 16px.
   Even at 3.6× magnification, a 20px Frigate streak → 72px at model input →
   4.5 patches. That is detectable but marginal. Need to run the experiment
   before claiming this fixes Frigate.

2. **Long-streak fragmentation** — A 1450px BrentImages streak in a 400px
   native tile always crosses multiple tiles. Cross-tile NMS merges detections
   when boxes overlap (IoU ≥ 0.4), but a streak spanning 3+ tiles may have
   non-overlapping fragments. Consider a second-pass "long-streak stitcher"
   that merges collinear boxes with low IoU but high directional alignment.

3. **Upscaling quality** — Bilinear upsampling of a 110px crop to 400px (3.6×)
   degrades high-frequency detail. LANCZOS or FITS-aware stretch + upscale may
   preserve more edge information for the ViT backbone.

4. **Overlap cost at 50% vs 25%** — BrentImages Night 2 at 50% overlap generates
   ~620 tiles per image and takes ~3.4 min/image on CPU. At 25% overlap that
   drops to ~185 tiles — 3× faster, at the cost of missing streaks near tile
   boundaries. For publication-quality eval, 50% is correct; for production API
   inference, 25% or a smarter "streak-aware overlap" (higher near image edges)
   may be warranted.

5. **Training data for Frigate regime** — Even with correct magnification at
   inference, the model was never trained on 3.6× upscaled crops. True fix
   requires adding tiled Frigate crops (with appropriate zoom) to the training
   set. See `scripts/build_tiled_frigate_json.py` — currently tiles at 1:1;
   needs a `native_tile_size` parameter to produce zoom-in crops.

---

## 5. Files to Create / Modify

| File | Action | Notes |
|------|--------|-------|
| `inference/tiled_pipeline.py` | **modify** | Add `native_tile_size`, `resize_to` to `tile_image()`; update `remap_predictions()` for magnification; update `run_tiled_inference()` signature |
| `inference/tiled_pipeline.py` | **add** | `select_tile_params()` helper |
| `scripts/build_tiled_frigate_json.py` | **modify** | Add `--native-tile-size` arg; generate zoom-in crops for training |
| `scripts/eval_brentimages_tiled.py` | **modify** | Add `--native-tile-size` arg; default stays 400 (BrentImages is already correct) |
| `scripts/eval_frigate_tiled.py` | **new** | Frigate-specific tiled eval using `native_tile_size=110` |

No changes to `inference/pipeline.py` or any model configs.

---

## 6. Verification Plan

1. **Frigate eval with `native_tile_size=110`** — run `scripts/eval_frigate_tiled.py`
   against `data/annotations/frigate_streaks_eval.json`; expect mAP@50 > 0.0
   (any improvement from current 0.000).

2. **BrentImages regression** — confirm `native_tile_size=400` still gives
   the same result as the current tiled eval (no regression from refactoring).

3. **SatStreaks regression** — full-image inference path (no tiling) unchanged;
   mAP@50 should remain 0.755.

4. **Training experiment** — retile Frigate crops at `native_tile_size=110`,
   add to `all_train_nodm.json`, run a 15-epoch fine-tune on top of the current
   best checkpoint; eval should show Frigate mAP@50 > 0.10.
