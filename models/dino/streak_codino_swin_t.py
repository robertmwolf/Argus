# Source: StreakMind — DINO Swin-T config for satellite streak detection
# Ref: agent_docs/streakmind_phases.md
#
# Swin-T (Tiny) variant for local Mac development and CI.
# Uses MODEL_SIZE=tiny.  Image size 400px, batch_size=1, num_workers=0.
#
# USAGE:  export MODEL_SIZE=tiny (default)
#         python -m inference.pipeline --fast --image data/raw/sample.fits
#
# IMPORTANT: This config is architecturally identical to streak_codino_swin_l.py
# but uses the Swin-T backbone (~28 M params vs ~197 M for Swin-L).
# Swin-T fits in 16 GB unified memory; Swin-L requires ≥22 GB VRAM.
#
# TRAINING:
#   Stage 1 (epochs 1-20):  backbone frozen   (lr_mult=0.0)
#   Stage 2 (epochs 21-50): backbone unfrozen (lr_mult=0.1)

# ---------------------------------------------------------------------------
# Custom FITS transform registration
# ---------------------------------------------------------------------------
# training.transforms registers LoadFITSFromFile with the mmcv TRANSFORMS
# registry so it can be referenced by name in the pipeline dicts below.
custom_imports = dict(imports=['training.transforms'], allow_failed_imports=False)

# ---------------------------------------------------------------------------
# Base dataset and runtime settings
# ---------------------------------------------------------------------------
dataset_type = 'CocoDataset'
data_root = 'data/'
backend_args = None

metainfo = dict(classes=('streak',), palette=[(220, 20, 60)])

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Source: DINO (Zong et al., 2023) — Swin-T backbone variant
# Ref: https://arxiv.org/abs/2203.03605
num_levels = 4      # Swin-T uses 4 feature levels (vs 5 for Swin-L)

model = dict(
    type='DINO',
    num_queries=300,        # reduced from 900 for memory efficiency on Mac
    with_box_refine=True,
    as_two_stage=True,

    # Source: StreakMind — Z-score normalisation stats for FITS images
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[51.0, 51.0, 51.0],
        bgr_to_rgb=False,
        pad_size_divisor=1,
    ),

    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,           # gradient checkpointing — required on 16 GB
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='weights/co_dino_swin_t_coco.pth',
        ),
    ),

    neck=dict(
        type='ChannelMapper',
        in_channels=[96, 192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels,
    ),

    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_levels=num_levels,
                dropout=0.0,
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0,
            ),
        ),
    ),

    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=num_levels,
                dropout=0.0,
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0,
            ),
        ),
        post_norm_cfg=None,
    ),

    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20,
    ),

    bbox_head=dict(
        type='DINOHead',
        num_classes=1,          # single class: 'streak'
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
    ),

    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
    ),

    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ],
        ),
    ),
    test_cfg=dict(max_per_img=100),
)

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
# Image size 256 — MPS has a 4GB NDArray limit per operation; DINO's
# deformable attention at larger sizes exceeds it on M3 16 GB.
_img_scale = (256, 256)

train_pipeline = [
    dict(type='LoadFITSFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoiceResize',
        scales=[_img_scale],
        keep_ratio=True,
    ),
    dict(type='PackDetInputs'),
]

val_pipeline = [
    dict(type='LoadFITSFromFile'),
    dict(type='Resize', scale=_img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'),
    ),
]

# ---------------------------------------------------------------------------
# Dataloaders  (Mac-safe values: batch_size=1, num_workers=0)
# ---------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=1,
    num_workers=0,          # MPS requires num_workers=0
    persistent_workers=False,
    pin_memory=False,       # not supported on MPS
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/dev_subset.json',   # 50-image dev subset
        data_prefix=dict(img='raw/'),  # COCO file_names use relative ../dev_subset/ paths
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/dev_subset.json',
        data_prefix=dict(img='raw/'),  # COCO file_names use relative ../dev_subset/ paths
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/dev_subset.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

# ---------------------------------------------------------------------------
# Optimiser and scheduler — two-stage fine-tuning
# ---------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-5,
        weight_decay=1e-4,
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0),   # Stage 1: frozen — set to 0.1 for Stage 2
            'neck':     dict(lr_mult=0.1),
            'encoder':  dict(lr_mult=1.0),
            'decoder':  dict(lr_mult=1.0),
            'bbox_head': dict(lr_mult=1.0),
        },
    ),
)

max_epochs = 50

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40, 47],
        gamma=0.1,
    ),
]

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),  # gloo for CPU/MPS; nccl for CUDA
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
log_level = 'INFO'
load_from = None          # set to 'weights/co_dino_swin_t_coco.pth' before training
resume = False

auto_scale_lr = dict(base_batch_size=16)
