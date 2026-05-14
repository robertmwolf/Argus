_img_scale = (
    256,
    256,
)
auto_scale_lr = dict(base_batch_size=16)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'training.transforms',
    ])
data_root = 'data/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=3,
        rule='greater',
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='gloo'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
load_from = 'weights/local_run/best_coco_bbox_mAP_epoch_50.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
max_epochs = 50
metainfo = dict(
    classes=('streak', ), palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        init_cfg=dict(
            checkpoint='weights/co_dino_swin_t_coco.pth', type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        pretrain_img_size=224,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=True),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=1,
        sync_cls_avg_factor=True,
        type='DINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        pad_size_divisor=1,
        std=[
            51.0,
            51.0,
            51.0,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=300,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=100),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DINO',
    with_box_refine=True)
num_levels = 4
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=1e-05, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.0),
            bbox_head=dict(lr_mult=1.0),
            decoder=dict(lr_mult=1.0),
            encoder=dict(lr_mult=1.0),
            neck=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=50,
        gamma=0.1,
        milestones=[
            40,
            47,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/dev_subset.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='data/',
        metainfo=dict(classes=('streak', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(type='LoadFITSFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/annotations/dev_subset.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    outfile_prefix=
    'results/phase_e/swin_t/best_coco_bbox_mAP_epoch_50_dev_subset',
    type='CocoMetric')
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        ann_file='annotations/dev_subset.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='data/',
        filter_cfg=dict(filter_empty_gt=False),
        metainfo=dict(classes=('streak', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(type='LoadFITSFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                keep_ratio=True,
                scales=[
                    (
                        256,
                        256,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadFITSFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(keep_ratio=True, scales=[
        (
            256,
            256,
        ),
    ], type='RandomChoiceResize'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/dev_subset.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='data/',
        metainfo=dict(classes=('streak', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(type='LoadFITSFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/annotations/dev_subset.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadFITSFromFile'),
    dict(keep_ratio=True, scale=(
        256,
        256,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'results/phase_e/swin_t'
