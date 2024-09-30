default_scope = 'mmpose'

# hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
)

# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]

# multi-processing backend
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# logger
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False

# file I/O backend
# backend_args = dict(backend='local')

# training/validation/testing progress
# train_cfg = dict(by_epoch=True)
val_cfg = dict()
# test_cfg = dict()

# 数据集类型及路径
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/Triangle_215_Keypoint_coco/'

# 三角板关键点检测数据集-元数据
dataset_info = {
    'dataset_name':'Triangle_215_Keypoint_coco',
    'classes':'sjb_rect',
    'paper_info':{
        'author':'Tongji Zihao',
        'title':'Triangle Keypoints Detection',
        'container':'OpenMMLab',
        'year':'2023',
        'homepage':'https://space.bilibili.com/1900783'
    },
    'keypoint_info':{
        0:{'name':'angle_30','id':0,'color':[255,0,0],'type': '','swap': ''},
        1:{'name':'angle_60','id':1,'color':[0,255,0],'type': '','swap': ''},
        2:{'name':'angle_90','id':2,'color':[0,0,255],'type': '','swap': ''}
    },
    'skeleton_info': {
        0: {'link':('angle_30','angle_60'),'id': 0,'color': [100,150,200]},
        1: {'link':('angle_60','angle_90'),'id': 1,'color': [200,100,150]},
        2: {'link':('angle_90','angle_30'),'id': 2,'color': [150,120,100]}
    },
    'joint_weights':[1.0, 1.0, 1.0],
    'sigmas':[0.026,0.025,0.025]
}

NUM_KEYPOINTS = len(dataset_info['keypoint_info'])

# 训练超参数
max_epochs = 300 # 训练 epoch 总数
val_interval = 10 # 每隔多少个 epoch 保存一次权重文件
train_cfg = {'max_epochs': max_epochs, 'val_interval': val_interval}
train_batch_size = 32
val_batch_size = 8
stage2_num_epochs = 0
base_lr = 4e-3
randomness = dict(seed=21)

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 学习率
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=20),
    dict(
        # use cosine lr from 210 to 420 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(12, 12),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# 不同输入图像尺寸的参数搭配
# input_size=(256, 256),
# sigma=(12, 12)
# in_featuremap_size=(8, 8)
# input_size可以换成 256、384、512、1024，三个参数等比例缩放
# sigma 表示关键点一维高斯分布的标准差，越大越容易学习，但精度上限会降低，越小越严格，对于人体、人脸等高精度场景，可以调小，RTMPose 原始论文中为 5.66

# 不同模型的 config： https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose/rtmpose/body_2d_keypoint

## 模型：RTMPose-S
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/' +
            'rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth',  # noqa E501
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=NUM_KEYPOINTS,
        input_size=codec['input_size'],
        in_featuremap_size=(8, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

## 模型：RTMPose-M
# model = dict(
#     type='TopdownPoseEstimator',
#     data_preprocessor=dict(
#         type='PoseDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True),
#     backbone=dict(
#         _scope_='mmdet',
#         type='CSPNeXt',
#         arch='P5',
#         expand_ratio=0.5,
#         deepen_factor=0.67,
#         widen_factor=0.75,
#         out_indices=(4, ),
#         channel_attention=True,
#         norm_cfg=dict(type='SyncBN'),
#         act_cfg=dict(type='SiLU'),
#         init_cfg=dict(
#             type='Pretrained',
#             prefix='backbone.',
#             checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'
#         )),
#     head=dict(
#         type='RTMCCHead',
#         in_channels=768,
#         out_channels=NUM_KEYPOINTS,
#         input_size=codec['input_size'],
#         in_featuremap_size=(8, 8),
#         simcc_split_ratio=codec['simcc_split_ratio'],
#         final_layer_kernel_size=7,
#         gau_cfg=dict(
#             hidden_dims=256,
#             s=128,
#             expansion_factor=2,
#             dropout_rate=0.,
#             drop_path=0.,
#             act_fn='SiLU',
#             use_rel_bias=False,
#             pos_enc=False),
#         loss=dict(
#             type='KLDiscretLoss',
#             use_target_weight=True,
#             beta=10.,
#             label_softmax=True),
#         decoder=codec),
#     test_cfg=dict(flip_test=True))

## 模型：RTMPose-L
# model = dict(
#     type='TopdownPoseEstimator',
#     data_preprocessor=dict(
#         type='PoseDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True),
#     backbone=dict(
#         _scope_='mmdet',
#         type='CSPNeXt',
#         arch='P5',
#         expand_ratio=0.5,
#         deepen_factor=1.,
#         widen_factor=1.,
#         out_indices=(4, ),
#         channel_attention=True,
#         norm_cfg=dict(type='SyncBN'),
#         act_cfg=dict(type='SiLU'),
#         init_cfg=dict(
#             type='Pretrained',
#             prefix='backbone.',
#             checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'
#         )),
#     head=dict(
#         type='RTMCCHead',
#         in_channels=1024,
#         out_channels=NUM_KEYPOINTS,
#         input_size=codec['input_size'],
#         in_featuremap_size=(8, 8),
#         simcc_split_ratio=codec['simcc_split_ratio'],
#         final_layer_kernel_size=7,
#         gau_cfg=dict(
#             hidden_dims=256,
#             s=128,
#             expansion_factor=2,
#             dropout_rate=0.,
#             drop_path=0.,
#             act_fn='SiLU',
#             use_rel_bias=False,
#             pos_enc=False),
#         loss=dict(
#             type='KLDiscretLoss',
#             use_target_weight=True,
#             beta=10.,
#             label_softmax=True),
#         decoder=codec),
#     test_cfg=dict(flip_test=True))


backend_args = dict(backend='local')
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         f'{data_root}': 's3://openmmlab/datasets/detection/coco/',
#         f'{data_root}': 's3://openmmlab/datasets/detection/coco/'
#     }))

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.8, 1.2], rotate_factor=30),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='ChannelShuffle', p=0.5),
            dict(type='CLAHE', p=0.5),
            # dict(type='Downscale', scale_min=0.7, scale_max=0.9, p=0.2),
            dict(type='ColorJitter', p=0.5),
            dict(
                type='CoarseDropout',
                max_holes=4,
                max_height=0.3,
                max_width=0.3,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
# train_pipeline = [
#     dict(type='LoadImage', backend_args=backend_args),
#     dict(type='GetBBoxCenterScale'),
#     dict(type='TopdownAffine', input_size=codec['input_size']),
#     dict(type='GenerateTarget', encoder=codec),
#     dict(type='PackPoseInputs')
# ]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='train_coco.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    # dataset=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     metainfo=dataset_info,
    #     data_mode=data_mode,
    #     ann_file='val_coco.json',
    #     # bbox_file=f'{data_root}person_detection_results/'
    #     # 'COCO_val2017_detections_AP_H_56_person.json',
    #     data_prefix=dict(img='images/'),
    #     test_mode=True,
    #     pipeline=val_pipeline,
    # )
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='val_coco.json',
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
# default_hooks = dict(
#     checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1),
#     logger=dict(interval=1),
# )

# default_hooks = {
#     'checkpoint': {'save_best': 'coco/AP','rule': 'greater','max_keep_ckpts': 1},
#     'logger': {'interval': 1}
# }

default_hooks = {
    'checkpoint': {'save_best': 'PCK','rule': 'greater','max_keep_ckpts': 2},
    'logger': {'interval': 1}
}

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
# val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'val_coco.json')
# val_evaluator = dict(type='PCKAccuracy')
# val_evaluator = [dict(type='CocoMetric', ann_file=data_root + 'val_coco.json'), dict(type='PCKAccuracy')]

# val_evaluator = [
#     dict(type='CocoMetric', ann_file=data_root + 'val_coco.json'),
#     dict(type='PCKAccuracy'),
#     dict(type='AUC'),
#     dict(type='NME', norm_mode='keypoint_distance')
# ]

val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + 'val_coco.json'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[1, 2])
]

test_evaluator = val_evaluator


