# Run 3 — Cold-start DM-free paper model
#
# This is the canonical config for the Run 3 paper training run (decided 2026-05-26).
# Identical to streak_dinov3_vitb_400px.py with two additions:
#   1. randomness = dict(seed=42, deterministic=True)  — reproducibility gate
#   2. load_from = None   — explicit cold start (no DM-exposed checkpoint)
#
# All other parameters are unchanged from Run 2.  The 15-epoch LR schedule is
# designed to be checkpointed and resumed across multiple sessions:
#
#   Night 1:  --max-epochs 3   (warmup completes ep2, 1 cosine epoch)   ~10–11h Mac
#   Night 2+: --resume         (epochs 4+, Ctrl-C in morning, repeat)    ~11h/3ep Mac
#
# Full invocation (night 1):
#
#   cd /path/to/Argus && conda activate satid
#   PYTORCH_ENABLE_MPS_FALLBACK=1 \
#   USE_DEV_SUBSET=false \
#   TRAIN_ANN_FILE=/Volumes/External/TrainingData/annotations/all_train_nodm_external_abs.json \
#   VAL_ANN_FILE=/Volumes/External/TrainingData/annotations/val_external_abs.json \
#   ARGUS_NORM=autostretch \
#   caffeinate -i \
#   python -m training.train_dino \
#       --config models/dino/streak_dinov3_vitb_400px_run3.py \
#       --work-dir weights/run3_cold_nodm \
#       --max-epochs 3 \
#       --val-interval 1 \
#       --checkpoint-interval 1
#
# Resume (subsequent nights):
#
#   PYTORCH_ENABLE_MPS_FALLBACK=1 \
#   USE_DEV_SUBSET=false \
#   TRAIN_ANN_FILE=/Volumes/External/TrainingData/annotations/all_train_nodm_external_abs.json \
#   VAL_ANN_FILE=/Volumes/External/TrainingData/annotations/val_external_abs.json \
#   ARGUS_NORM=autostretch \
#   caffeinate -i \
#   python -m training.train_dino \
#       --config models/dino/streak_dinov3_vitb_400px_run3.py \
#       --work-dir weights/run3_cold_nodm \
#       --resume \
#       --val-interval 1 \
#       --checkpoint-interval 1
#
# Dataset: all_train_nodm.json v2 — 8,422 images, 8,213 annotations
#   SatStreaks 2,488 + BrentImages N1 3,110 (tiled 400px) +
#   BrentImages N2 1,309 (tiled 400px) + Frigate 1,515 (tiled 110px/3.64×)
#   Requires /Volumes/External/TrainingData mounted. Canonical raw images live in
#   /Volumes/External/TrainingData/raw and canonical annotations live in
#   /Volumes/External/TrainingData/annotations. data/ paths are compatibility shims.
#
# Checkpoint destination: weights/run3_cold_nodm/
# Hardware: Mac M3 CPU (PYTORCH_ENABLE_MPS_FALLBACK=1)

custom_imports = dict(
    imports=['training.transforms', 'models.dino.dinov3_adapter'],
    allow_failed_imports=False,
)

dataset_type = 'CocoDataset'
data_root = 'data/'
backend_args = None

metainfo = dict(classes=('streak',), palette=[(220, 20, 60)])

num_levels = 4

model = dict(
    type='DINO',
    num_queries=300,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.280, 103.530],
        std=[58.395, 57.120, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=16,
    ),
    backbone=dict(
        type='DINOv3Backbone',
        model_size='base',
        weights='weights/dinov3_vitb16_lvd1689m.pth',
        frozen=True,
        out_channels=768,
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[768, 768, 768, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels,
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=num_levels, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
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
# Data pipeline — 400px (paper resolution, unchanged from Run 2)
# ---------------------------------------------------------------------------
_img_scale = (400, 400)

train_pipeline = [
    dict(type='LoadFITSFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoiceResize', scales=[_img_scale], keep_ratio=True),
    dict(type='PackDetInputs'),
]

val_pipeline = [
    dict(type='LoadFITSFromFile'),
    dict(type='Resize', scale=_img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',          # overridden by TRAIN_ANN_FILE env var
        data_prefix=dict(img=''),
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
        ann_file='annotations/val.json',            # overridden by VAL_ANN_FILE env var
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone':  dict(lr_mult=0.0),
            'neck':      dict(lr_mult=1.0),
            'encoder':   dict(lr_mult=1.0),
            'decoder':   dict(lr_mult=1.0),
            'bbox_head': dict(lr_mult=1.0),
        },
    ),
)

max_epochs = 15

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,                  # override with --val-interval 1 for overnight runs
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=2,
        convert_to_iter_based=False,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=13,
        eta_min=1e-6,
        begin=2,
        end=15,
        by_epoch=True,
        convert_to_iter_based=False,
    ),
]

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,                  # override with --checkpoint-interval 1 for overnight runs
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
    dist_cfg=dict(backend='gloo'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
log_level = 'INFO'

# ---------------------------------------------------------------------------
# Paper run settings — do not change
# ---------------------------------------------------------------------------
load_from = None   # Cold start: detection head initialised from scratch.
                   # No DM-exposed checkpoint in the initialisation chain.
                   # See docs/training_methods.md §3.1 for rationale.

resume = False

# Source: docs/training_methods.md §4 Paper Run Config Parameters
# "randomness = dict(seed=42, deterministic=True)" — reproducibility requirement
randomness = dict(seed=42, deterministic=True)
