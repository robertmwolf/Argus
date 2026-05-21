from __future__ import annotations

import numpy as np

from inference.tiled_pipeline import nms_predictions, remap_predictions, tile_image


def test_tile_image_pads_and_covers_edges() -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)

    tiles = list(tile_image(image, tile_size=4, overlap=0.5))

    assert [(x0, y0) for _, x0, y0 in tiles] == [
        (0, 0), (2, 0), (4, 0),
        (0, 2), (2, 2), (4, 2),
    ]
    assert all(tile.shape == (4, 4, 3) for tile, _, _ in tiles)
    assert np.array_equal(tiles[-1][0][-1, -1], image[-1, -1])


def test_remap_predictions_offsets_xywh_boxes() -> None:
    preds = [{"bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9, "category_id": 1}]

    remapped = remap_predictions(preds, x0=10, y0=20)

    assert remapped == [{"bbox": [11.0, 22.0, 3.0, 4.0], "score": 0.9, "category_id": 1}]
    assert preds[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_nms_predictions_suppresses_same_category_duplicates() -> None:
    preds = [
        {"bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9, "category_id": 1},
        {"bbox": [1.0, 1.0, 10.0, 10.0], "score": 0.8, "category_id": 1},
        {"bbox": [1.0, 1.0, 10.0, 10.0], "score": 0.7, "category_id": 2},
        {"bbox": [30.0, 30.0, 5.0, 5.0], "score": 0.6, "category_id": 1},
    ]

    kept = nms_predictions(preds, iou_threshold=0.4)

    assert kept == [preds[0], preds[2], preds[3]]
