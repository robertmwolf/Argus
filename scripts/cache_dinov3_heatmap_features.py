"""Cache frozen DINOv3 features for the plain heatmap spike."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.device import get_device
from models.plain_dinov3.streak_heatmap import DINOv3StreakHeatmap, imagenet_normalize
from training.dinov3_heatmap_dataset import StreakHeatmapDataset, collate_heatmap_batch

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", default="weights/dinov3_vitb16_lvd1689m.pth")
    parser.add_argument("--model-size", choices=["base", "large"], default="base")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    out_dir = Path(args.output_dir)
    feature_dir = out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    ds = StreakHeatmapDataset(args.annotations, image_size=args.image_size, max_samples=args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_heatmap_batch,
    )
    model = DINOv3StreakHeatmap(model_size=args.model_size, weights=args.weights).to(device)
    model.eval()

    manifest: list[dict] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = imagenet_normalize(batch["image"].to(device))
            features = model.extract_features(images).cpu().to(torch.float16)
            heatmaps = batch["heatmap"].cpu().to(torch.float16)
            center_heatmaps = batch["center_heatmap"].cpu().to(torch.float16)
            box_targets = batch["box_target"].cpu().to(torch.float16)
            box_masks = batch["box_mask"].cpu().to(torch.float16)
            geometries = batch["geometry"].cpu().to(torch.float16)
            image_ids = batch["image_id"].cpu().tolist()
            orig_sizes = batch["orig_size"].cpu().tolist()
            letterboxes = batch["letterbox"].cpu().tolist()
            file_names = batch["file_name"]

            for i, image_id in enumerate(image_ids):
                rel_path = Path("features") / f"{int(image_id)}.pt"
                torch.save(
                    {
                        "features": features[i],
                        "heatmap": heatmaps[i],
                        "center_heatmap": center_heatmaps[i],
                        "box_target": box_targets[i],
                        "box_mask": box_masks[i],
                        "geometry": geometries[i],
                        "image_id": int(image_id),
                        "orig_size": orig_sizes[i],
                        "letterbox": letterboxes[i],
                        "file_name": file_names[i],
                    },
                    out_dir / rel_path,
                )
                manifest.append({
                    "image_id": int(image_id),
                    "path": str(rel_path),
                    "file_name": file_names[i],
                })
            logger.info("cached batch %d/%d (%d samples)", batch_idx, len(loader), len(manifest))

    metadata = {
        "annotations": args.annotations,
        "weights": args.weights,
        "model_size": args.model_size,
        "image_size": args.image_size,
        "n_samples": len(manifest),
        "manifest": manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2))
    logger.info("wrote cache manifest: %s", out_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
