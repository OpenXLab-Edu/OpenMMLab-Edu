# mask_rcnn_regnetx-3.2GF_fpn_1x_coco_20200520_163141-2a9d1814.pth
# model settings
model = dict(backbone=dict(_delete_=True,
                           type='RegNet',
                           arch='regnetx_3.2gf',
                           out_indices=(0, 1, 2, 3),
                           frozen_stages=1,
                           norm_cfg=dict(type='BN', requires_grad=True),
                           norm_eval=True,
                           style='pytorch',
                           init_cfg=dict(
                               type='Pretrained',
                               checkpoint='open-mmlab://regnetx_3.2gf')),
             neck=dict(type='FPN',
                       in_channels=[96, 192, 432, 1008],
                       out_channels=256,
                       num_outs=5))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    # The mean and std are used in PyCls when training RegNets
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    to_rgb=False)
train_pipeline = [
    # Images are converted to float32 directly after loading in PyCls
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1333, 800),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]
data = dict(train=dict(pipeline=train_pipeline),
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
