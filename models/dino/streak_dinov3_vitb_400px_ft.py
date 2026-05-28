# Fine-tune config — DINOv3 ViT-B, 400px, from Run 3 checkpoint
#
# Use this config when adding a new telescope's data to an already-trained
# model.  Key differences from streak_dinov3_vitb_400px_run3.py:
#
#   1. load_from points to the Run 3 best checkpoint (stable symlink).
#      The backbone is frozen (lr_mult=0.0) — we only update the Co-DINO
#      detection head.  This avoids catastrophic forgetting of backbone
#      features while adapting the head to the new domain.
#
#   2. Base LR is 10× lower (1e-5 vs 1e-4).  Pre-trained weights are
#      already near a good minimum; large LR would destroy them.
#
#   3. No warmup — warmup is only needed for cold starts.  The LR schedule
#      starts directly with a short cosine anneal.
#
#   4. max_epochs = 8.  Fine-tuning converges faster than cold-start training.
#      Extend to 12 if loss has not plateaued at epoch 8.
#
# Training mix recommendation
# ---------------------------
# To avoid the detection head forgetting existing-domain performance, always
# include existing-scope data alongside the new-scope data in the training
# JSON.  A 1:1 ratio is a safe default:
#
#   python scripts/build_training_json.py \
#       --output data/annotations/all_train_ft_newscope.json
#   # (uses manifest mix_weight values; set newscope mix_weight to match
#   #  existing scope image count for a 1:1 ratio, or use --mix-ratio)
#
# Full invocation
# ---------------
#   cd /path/to/Argus && conda activate satid
#   PYTORCH_ENABLE_MPS_FALLBACK=1 \
#   USE_DEV_SUBSET=false \
#   TRAIN_ANN_FILE=annotations/all_train_ft_newscope.json \
#   VAL_ANN_FILE=annotations/val.json \
#   ARGUS_NORM=autostretch \
#   caffeinate -i \
#   python -m training.train_dino \
#       --config models/dino/streak_dinov3_vitb_400px_ft.py \
#       --work-dir weights/run4_ft_newscope \
#       --max-epochs 8 \
#       --val-interval 1 \
#       --checkpoint-interval 1
#
# Evaluation after fine-tuning
# ----------------------------
# Must pass BOTH of these checks before promoting:
#   1. New scope Night 1 (holdout): recall ≥ (zero_shot_recall + 5pp)
#   2. Standard test set (test.json): recall ≥ (Run 3 recall - 2pp)
#      i.e., the fine-tune must not hurt existing-domain performance by
#      more than 2 percentage points.
#
#   python scripts/evaluate_comprehensive.py \
#       --checkpoint weights/run4_ft_newscope/best.pth \
#       --config models/dino/streak_dinov3_vitb_400px_ft.py

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
        frozen=True,           # Backbone stays frozen throughout fine-tuning.
        out_channels=768,      # Keeps learned features intact.
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
# Data pipeline — 400px (identical to run3)
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

# ---------------------------------------------------------------------------
# Optimiser — 10× lower LR than cold-start run3
# ---------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-5,            # 10× lower than Run 3 (1e-4); head is already warm
        weight_decay=1e-4,
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone':  dict(lr_mult=0.0),   # frozen — receives zero gradient
            'neck':      dict(lr_mult=1.0),
            'encoder':   dict(lr_mult=1.0),
            'decoder':   dict(lr_mult=1.0),
            'bbox_head': dict(lr_mult=1.0),
        },
    ),
)

max_epochs = 8   # Fine-tuning converges faster; extend to 12 if not plateaued

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,           # validate every epoch (fine-tune runs are short)
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# No warmup: the weights are already trained.  Short cosine anneal from
# base LR (1e-5) to near-zero (1e-7) over the full fine-tune window.
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        eta_min=1e-7,
        begin=0,
        end=max_epochs,
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
        interval=1,
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
# Checkpoint initialisation — Run 3 best (stable symlink)
# ---------------------------------------------------------------------------
# Update this path to point at whichever run you are fine-tuning from.
# The stable ``best.pth`` symlink always points to the current best epoch
# inside the work dir (managed by CheckpointHook during training).
load_from = "weights/run3_cold_nodm/best.pth"

resume = False

# Keep the same seed for reproducibility of the fine-tune run.
randomness = dict(seed=42, deterministic=True)
