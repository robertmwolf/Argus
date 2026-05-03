# Source: StreakMind — Co-DINO Swin-L config for satellite streak detection
# Ref: agent_docs/streakmind_phases.md
#
# Adapted from MMDetection DINO Swin-L config:
#   mmdet/.mim/configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py
# Ref: https://arxiv.org/abs/2203.03605
#
# IMPORTANT: Co-DINO (Co-Deformable DETR) is not in the mmdet 3.3.0 pip
# release. This config uses the DINO detector, which is the transformer
# backbone of Co-DINO. For cloud training, the model type and head structure
# can be upgraded to full Co-DINO if the package is installed from source.
#
# USAGE: Cloud training only — requires CUDA and MODEL_SIZE=large.
# For local Mac development use streak_codino_swin_t.py instead.
#
# TRAINING:
#   Stage 1 (epochs 1-20):  backbone frozen   (lr_mult=0.0)
#   Stage 2 (epochs 21-50): backbone unfrozen (lr_mult=0.1)
#
# See training/train_dino.py for the two-stage schedule implementation.

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
# Source: DINO (Zong et al., 2023) — Swin-L backbone config
# Ref: https://arxiv.org/abs/2203.03605
num_levels = 5

model = dict(
    type='DINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,

    # Source: StreakMind — Z-score normalisation stats for FITS images
    # Our FITSLoader clips to ±3σ then maps to [0, 255] uint8 (3-channel).
    # These mean/std values match that Z-score normalisation:
    #   mean ≈ 127.5 (midpoint of 0–255)
    #   std  ≈ 51.0  (255 / (2 × 2.5) — conservative ±2.5σ effective range)
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[51.0, 51.0, 51.0],
        bgr_to_rgb=False,   # our FITS images are greyscale stacked, not BGR
        pad_size_divisor=1,
    ),

    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,           # gradient checkpointing — required at 40 GB
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='weights/co_dino_swin_l_coco.pth',
        ),
    ),

    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768, 1536],
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
    test_cfg=dict(max_per_img=300),
)

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 800), (512, 800), (544, 800), (576, 800),
                            (608, 800), (640, 800), (672, 800), (704, 800),
                            (736, 800), (768, 800), (800, 800)],
                    keep_ratio=True,
                ),
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 800), (500, 800), (600, 800)],
                    keep_ratio=True,
                ),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 800), (512, 800), (544, 800), (576, 800),
                            (608, 800), (640, 800), (672, 800), (704, 800),
                            (736, 800), (768, 800), (800, 800)],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type='PackDetInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'),
    ),
]

# ---------------------------------------------------------------------------
# Dataloaders  (cloud training values — A100 40 GB)
# ---------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,        # safe on CUDA; improves host→GPU transfer speed
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='raw/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='raw/'),
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
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

# ---------------------------------------------------------------------------
# Optimiser and scheduler — two-stage fine-tuning
# ---------------------------------------------------------------------------
# Source: StreakMind — two-stage fine-tuning strategy
# Stage 1 (epochs 1-20):  backbone frozen,   only neck + head train
# Stage 2 (epochs 21-50): backbone unfrozen, lower LR for backbone
#
# The training script (training/train_dino.py) switches lr_mult for the
# backbone from 0.0 → 0.1 at epoch 21 by updating this config programmatically.

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
    logger=dict(type='LoggerHook', interval=50),
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
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None          # set to 'weights/co_dino_swin_l_coco.pth' before training
resume = False

auto_scale_lr = dict(base_batch_size=16)
