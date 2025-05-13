auto_scale_lr = dict(base_batch_size=256)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=5,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(_scope_='mmcls', interval=5, type='CheckpointHook'),
    logger=dict(_scope_='mmcls', interval=100, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmcls', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmcls', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmcls', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmcls', enable=False, type='VisualizationHook'))
default_scope = 'mmcls'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
log_level = 'INFO'
model = dict(
    _scope_='mmcls',
    backbone=dict(
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=2048,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=5,
        topk=(1, ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(
        _scope_='mmcls',
        lr=0.001,
        momentum=0.9,
        type='SGD',
        weight_decay=0.0001))
param_scheduler = dict(
    _scope_='mmcls',
    by_epoch=True,
    gamma=0.1,
    milestones=[
        5,
        10,
        15,
    ],
    type='MultiStepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        _scope_='mmcls',
        ann_file='flower_dataset_processed/val.txt',
        classes='flower_dataset_processed/classes.txt',
        data_prefix='flower_dataset_processed/val',
        data_root='D:/Material/大三/神经网络与深度学习/Ex1',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmcls', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmcls', topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(_scope_='mmcls', type='LoadImageFromFile'),
    dict(_scope_='mmcls', edge='short', scale=256, type='ResizeEdge'),
    dict(_scope_='mmcls', crop_size=224, type='CenterCrop'),
    dict(_scope_='mmcls', type='PackClsInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        _scope_='mmcls',
        ann_file='flower_dataset_processed/train.txt',
        classes='flower_dataset_processed/classes.txt',
        data_prefix='flower_dataset_processed/train',
        data_root='D:/Material/大三/神经网络与深度学习/Ex1',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmcls', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(_scope_='mmcls', type='LoadImageFromFile'),
    dict(_scope_='mmcls', scale=224, type='RandomResizedCrop'),
    dict(_scope_='mmcls', direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(_scope_='mmcls', type='PackClsInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        _scope_='mmcls',
        ann_file='flower_dataset_processed/val.txt',
        classes='flower_dataset_processed/classes.txt',
        data_prefix='flower_dataset_processed/val',
        data_root='D:/Material/大三/神经网络与深度学习/Ex1',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmcls', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(_scope_='mmcls', topk=(1, ), type='Accuracy')
vis_backends = [
    dict(_scope_='mmcls', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmcls',
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dir/resnet50_finetune_flower'
