model = dict(
    type='BasicGaussianDiffusion',
    num_timesteps=4000,
    betas_cfg=dict(type='cosine'),
    denoising=dict(
        type='DenoisingUnet',
        image_size=32,
        in_channels=3,
        base_channels=128,
        resblocks_per_downsample=3,
        attention_res=[16, 8],
        use_scale_shift_norm=True,
        dropout=0.3,
        num_heads=4,
        use_rescale_timesteps=True,
        output_cfg=dict(mean='eps', var='learned_range')),
    timestep_sampler=dict(type='UniformTimeStepSampler'),
    ddpm_loss=[
        dict(
            type='DDPMVLBLoss',
            rescale_mode='constant',
            rescale_cfg=dict(scale=4000 / 1000),
            data_info=dict(
                mean_pred='mean_pred',
                mean_target='mean_posterior',
                logvar_pred='logvar_pred',
                logvar_target='logvar_posterior'),
            log_cfgs=[
                dict(
                    type='quartile',
                    prefix_name='loss_vlb',
                    total_timesteps=4000),
                dict(type='name')
            ]),
        dict(
            type='DDPMMSELoss',
            log_cfgs=dict(
                type='quartile', prefix_name='loss_mse', total_timesteps=4000),
        )
    ],
)

train_cfg = dict(use_ema=True, real_img_key='img')
test_cfg = None
optimizer = dict(denoising=dict(type='AdamW', lr=1e-4, weight_decay=0))

dataset_type = 'mmcls.CIFAR10'

# different from mmcls, we adopt the setting used in BigGAN
# Note that the pipelines below are from MMClassification. Importantly, the
# `to_rgb` is set to `True` to convert image to BGR orders. The default order
# in Cifar10 is RGB. Thus, we have to convert it to BGR.

# Cifar dataset w/o augmentations. Remove `RandomFlip` and `RandomCrop`
# augmentations.
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# Different from the classification task, the val/test split also use the
# training part, which is the same to StyleGAN-ADA.

# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'


lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=1000),
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=['real_imgs', 'x_0_pred', 'x_t', 'x_t_1'],
        padding=1,
        interval=1000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('denoising_ema'),
        interval=1,
        start_iter=0,
        interp_cfg=dict(momentum=0.9999),
        priority='VERY_HIGH')
]

# do not evaluation in training process because evaluation take too much time.
evaluation = None

total_iters = 500000  # 500k

data = dict(
    samples_per_gpu=16,  # 8x16=128
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_prefix='data/cifar10',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type, data_prefix='data/cifar10', pipeline=test_pipeline),
    test=dict(
        type=dataset_type, data_prefix='data/cifar10', pipeline=test_pipeline))

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

inception_pkl = './work_dirs/inception_pkl/cifar10.pkl'
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        bgr2rgb=True,
        inception_pkl=inception_pkl,
        inception_args=dict(type='StyleGAN')))

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 10000)]
cudnn_benchmark = True