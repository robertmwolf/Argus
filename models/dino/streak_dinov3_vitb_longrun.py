# DINOv3 ViT-B/16 — 50-epoch long-run config for A/B comparison (256px)
#
# Differences from streak_dinov3_vitb.py:
#   - max_epochs = 50
#   - param_scheduler: 2-epoch linear warmup then CosineAnnealingLR (1e-4 → 1e-6)
#   - val_interval = 5, checkpoint interval = 5, max_keep_ckpts = 3
#   - Image scale unchanged at 256px (keeps A/B runs fast: ~2.5h each)
#
# Use streak_dinov3_vitb_400px.py for the final quality run at 400px.
#
# USAGE (via train_overnight.sh or manually):
#   USE_DEV_SUBSET=false \
#   TRAIN_ANN_FILE=annotations/all_train_nodm.json \
#   VAL_ANN_FILE=annotations/dm_merged_val.json \
#   PYTORCH_ENABLE_MPS_FALLBACK=1 \
#   python -m training.train_dino \
#     --config models/dino/streak_dinov3_vitb_longrun.py \
#     --work-dir weights/run_50ep_nodm \
#     --load-from weights/run_gt_dm_satstreaks_dinov3_vitb/best_coco_bbox_mAP_epoch_4.pth

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
# Data pipeline — 256px (A/B comparison scale)
# ---------------------------------------------------------------------------
_img_scale = (256, 256)

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
        ann_file='annotations/train.json',
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
        ann_file='annotations/val.json',
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

# ---------------------------------------------------------------------------
# Optimiser — backbone lr_mult=0.0 permanently (frozen DINOv3)
# ---------------------------------------------------------------------------
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
    val_interval=5,
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 2-epoch linear warmup (1e-6 → 1e-4) then cosine anneal (1e-4 → 1e-6) over 13 epochs.
# The warmup protects the warm-started weights from a sudden LR spike.
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
    dist_cfg=dict(backend='gloo'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
