"""Tests for MMDetection config files in models/dino/.

Validates that both configs:
- parse without errors
- contain correct single-class streak settings
- have device-appropriate batch sizes / num_workers
- use Z-score normalisation statistics for FITS images
- define the two-stage backbone LR schedule
"""

from __future__ import annotations

import pytest

try:
    from mmengine.config import Config
    import mmdet.models  # trigger DINO model registration  # noqa: F401
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MMDET_AVAILABLE, reason="mmdet / mmengine not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def swin_t_cfg():
    return Config.fromfile("models/dino/streak_codino_swin_t.py")


@pytest.fixture(scope="module")
def swin_l_cfg():
    return Config.fromfile("models/dino/streak_codino_swin_l.py")


# ---------------------------------------------------------------------------
# Shared assertions applied to both configs
# ---------------------------------------------------------------------------

class TestBothConfigs:
    @pytest.mark.parametrize("cfg_path", [
        "models/dino/streak_codino_swin_t.py",
        "models/dino/streak_codino_swin_l.py",
    ])
    def test_loads_without_error(self, cfg_path):
        cfg = Config.fromfile(cfg_path)
        assert cfg is not None

    @pytest.mark.parametrize("cfg_path", [
        "models/dino/streak_codino_swin_t.py",
        "models/dino/streak_codino_swin_l.py",
    ])
    def test_model_type_is_dino(self, cfg_path):
        cfg = Config.fromfile(cfg_path)
        assert cfg.model.type == "DINO"

    @pytest.mark.parametrize("cfg_path", [
        "models/dino/streak_codino_swin_t.py",
        "models/dino/streak_codino_swin_l.py",
    ])
    def test_backbone_is_swin(self, cfg_path):
        cfg = Config.fromfile(cfg_path)
        assert cfg.model.backbone.type == "SwinTransformer"

    @pytest.mark.parametrize("cfg_path", [
        "models/dino/streak_codino_swin_t.py",
        "models/dino/streak_codino_swin_l.py",
    ])
    def test_single_class_streak(self, cfg_path):
        """Both configs must detect exactly one class: 'streak'."""
        cfg = Config.fromfile(cfg_path)
        assert cfg.model.bbox_head.num_classes == 1
        assert cfg.metainfo.classes == ("streak",)

    @pytest.mark.parametrize("cfg_path", [
        "models/dino/streak_codino_swin_t.py",
        "models/dino/streak_codino_swin_l.py",
    ])
    def test_fits_zscore_normalisation(self, cfg_path):
        """data_preprocessor must use Z-score stats for FITS images."""
        cfg = Config.fromfile(cfg_path)
        dp = cfg.model.data_preprocessor
        # Mean should be ~127.5 (midpoint of 0-255 range after Z-score clip)
        assert dp.mean == [127.5, 127.5, 127.5], f"Wrong mean in {cfg_path}"
        assert dp.std == [51.0, 51.0, 51.0], f"Wrong std in {cfg_path}"
        # FITS images are stacked greyscale, not BGR from camera
        assert dp.bgr_to_rgb is False, "bgr_to_rgb must be False for FITS images"

    @pytest.mark.parametrize("cfg_path", [
        "models/dino/streak_codino_swin_t.py",
        "models/dino/streak_codino_swin_l.py",
    ])
    def test_max_epochs_50(self, cfg_path):
        cfg = Config.fromfile(cfg_path)
        assert cfg.train_cfg.max_epochs == 50

    @pytest.mark.parametrize("cfg_path", [
        "models/dino/streak_codino_swin_t.py",
        "models/dino/streak_codino_swin_l.py",
    ])
    def test_backbone_frozen_stage1(self, cfg_path):
        """Stage 1: backbone lr_mult must be 0.0 (frozen)."""
        cfg = Config.fromfile(cfg_path)
        custom_keys = cfg.optim_wrapper.paramwise_cfg.custom_keys
        assert "backbone" in custom_keys
        assert custom_keys["backbone"]["lr_mult"] == 0.0, (
            f"backbone lr_mult should be 0.0 in {cfg_path} (Stage 1 frozen)"
        )

    @pytest.mark.parametrize("cfg_path", [
        "models/dino/streak_codino_swin_t.py",
        "models/dino/streak_codino_swin_l.py",
    ])
    def test_gradient_checkpointing_enabled(self, cfg_path):
        cfg = Config.fromfile(cfg_path)
        assert cfg.model.backbone.with_cp is True, (
            "Gradient checkpointing (with_cp) must be True to fit in memory"
        )

    @pytest.mark.parametrize("cfg_path", [
        "models/dino/streak_codino_swin_t.py",
        "models/dino/streak_codino_swin_l.py",
    ])
    def test_with_box_refine_and_two_stage(self, cfg_path):
        cfg = Config.fromfile(cfg_path)
        assert cfg.model.with_box_refine is True
        assert cfg.model.as_two_stage is True

    @pytest.mark.parametrize("cfg_path", [
        "models/dino/streak_codino_swin_t.py",
        "models/dino/streak_codino_swin_l.py",
    ])
    def test_hungarian_assigner(self, cfg_path):
        cfg = Config.fromfile(cfg_path)
        assigner = cfg.model.train_cfg.assigner
        assert assigner.type == "HungarianAssigner"


# ---------------------------------------------------------------------------
# Swin-T specific (Mac dev config)
# ---------------------------------------------------------------------------

class TestSwinTConfig:
    def test_embed_dims_tiny(self, swin_t_cfg):
        assert swin_t_cfg.model.backbone.embed_dims == 96

    def test_depths_tiny(self, swin_t_cfg):
        assert swin_t_cfg.model.backbone.depths == [2, 2, 6, 2]

    def test_batch_size_1(self, swin_t_cfg):
        """Mac config must use batch_size=1."""
        assert swin_t_cfg.train_dataloader.batch_size == 1

    def test_num_workers_0(self, swin_t_cfg):
        """MPS requires num_workers=0."""
        assert swin_t_cfg.train_dataloader.num_workers == 0

    def test_pin_memory_false(self, swin_t_cfg):
        """pin_memory not supported on MPS."""
        assert swin_t_cfg.train_dataloader.pin_memory is False

    def test_num_queries_reduced(self, swin_t_cfg):
        """Swin-T uses 300 queries (vs 900) to fit in 16 GB."""
        assert swin_t_cfg.model.num_queries == 300

    def test_4_feature_levels(self, swin_t_cfg):
        assert swin_t_cfg.model.neck.num_outs == 4

    def test_dev_subset_annotation(self, swin_t_cfg):
        """Swin-T config should use the 50-image dev subset."""
        ann_file = swin_t_cfg.train_dataloader.dataset.ann_file
        assert "dev_subset" in ann_file, (
            "Swin-T config should use dev_subset.json for local development"
        )

    def test_window_size_7(self, swin_t_cfg):
        assert swin_t_cfg.model.backbone.window_size == 7


# ---------------------------------------------------------------------------
# Swin-L specific (cloud training config)
# ---------------------------------------------------------------------------

class TestSwinLConfig:
    def test_embed_dims_large(self, swin_l_cfg):
        assert swin_l_cfg.model.backbone.embed_dims == 192

    def test_depths_large(self, swin_l_cfg):
        assert swin_l_cfg.model.backbone.depths == [2, 2, 18, 2]

    def test_batch_size_1(self, swin_l_cfg):
        """Cloud config uses batch_size=1 with gradient accumulation for 16 GB VRAM."""
        assert swin_l_cfg.train_dataloader.batch_size == 1

    def test_grad_accumulation(self, swin_l_cfg):
        """Gradient accumulation=2 gives effective batch size=2."""
        assert swin_l_cfg.optim_wrapper.accumulative_counts == 2

    def test_num_workers_4(self, swin_l_cfg):
        assert swin_l_cfg.train_dataloader.num_workers == 4

    def test_pin_memory_true(self, swin_l_cfg):
        assert swin_l_cfg.train_dataloader.pin_memory is True

    def test_num_queries_900(self, swin_l_cfg):
        assert swin_l_cfg.model.num_queries == 900

    def test_5_feature_levels(self, swin_l_cfg):
        assert swin_l_cfg.model.neck.num_outs == 5

    def test_window_size_12(self, swin_l_cfg):
        assert swin_l_cfg.model.backbone.window_size == 12

    def test_train_annotation_full(self, swin_l_cfg):
        """Swin-L config should use the full train.json annotation."""
        ann_file = swin_l_cfg.train_dataloader.dataset.ann_file
        assert "train.json" in ann_file, (
            "Swin-L config should use full train.json for cloud training"
        )
