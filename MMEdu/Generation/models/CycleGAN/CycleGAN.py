domain_a = 'summer'
domain_b = 'winter'

model = dict(
    type='CycleGAN',
    generator=dict(
        type='ResnetGenerator',
        in_channels=3,
        out_channels=3,
        base_channels=64,
        norm_cfg=dict(type='IN'),
        use_dropout=False,
        num_blocks=9,
        padding_mode='reflect',
        init_cfg=dict(type='normal', gain=0.02)),
    discriminator=dict(
        type='PatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='IN'),
        init_cfg=dict(type='normal', gain=0.02)),
    gan_loss=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
    default_domain=None,  # set by user
    reachable_domains=None,  # set by user
    related_domains=None,  # set by user
    default_domain=domain_b,
    reachable_domains=[domain_a, domain_b],
    related_domains=[domain_a, domain_b],
    gen_auxiliary_loss=[
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{domain_a}', target=f'real_{domain_a}'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{domain_b}',
                target=f'real_{domain_b}',
            ),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(
                pred=f'identity_{domain_a}', target=f'real_{domain_a}'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(
                pred=f'identity_{domain_b}', target=f'real_{domain_b}'),
            reduction='mean')
    ])

train_cfg = dict(buffer_size=50)
test_cfg = None

train_dataset_type = 'UnpairedImageDataset'
val_dataset_type = 'UnpairedImageDataset'

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_a}',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_b}',
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(
        type='Crop',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        crop_size=(256, 256),
        random_crop=True),
    dict(type='Flip', keys=[f'img_{domain_a}'], direction='horizontal'),
    dict(type='Flip', keys=[f'img_{domain_b}'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Normalize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Collect',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_a}',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_b}',
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        scale=(256, 256),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Normalize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Collect',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]
dataroot = './data/summer2winter_yosemite'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=True,
    val_samples_per_gpu=1,
    val_workers_per_gpu=0,
    train=dict(
        type=train_dataset_type,
        dataroot=dataroot,
        pipeline=train_pipeline,
        test_mode=True,
        domain_a=domain_a,
        domain_b=domain_b),
    val=dict(
        type=val_dataset_type,
        dataroot=dataroot,
        test_mode=True,
        domain_a=domain_a,
        domain_b=domain_b,
        pipeline=test_pipeline),
    test=dict(
        type=val_dataset_type,
        dataroot=dataroot,
        test_mode=True,
        domain_a=domain_a,
        domain_b=domain_b,
        pipeline=test_pipeline))

optimizer = dict(
    generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(
    policy='Linear', by_epoch=False, target_lr=0, start=125000, interval=1250)

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=[f'fake_{domain_a}', f'fake_{domain_b}'],
        interval=5000)
]

# runner = None
# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=True,
    pass_training_status=True)
    
use_ddp_wrapper = True
total_iters = 250000
workflow = [('train', 1)]
exp_name = 'cyclegan_summer2winter'
work_dir = f'./work_dirs/experiments/{exp_name}'

# testA: 309, testB:238
num_images = 238
metrics = dict(
    FID=dict(type='FID', num_images=num_images, image_shape=(3, 256, 256)),
    IS=dict(
        type='IS',
        num_images=num_images,
        image_shape=(3, 256, 256),
        inception_args=dict(type='pytorch')))

evaluation = dict(
    type='TranslationEvalHook',
    target_domain=domain_b,
    interval=10000,
    metrics=[
        dict(type='FID', num_images=num_images, bgr2rgb=True),
        dict(
            type='IS',
            num_images=num_images,
            inception_args=dict(type='pytorch'))
    ],
    best_metric=['fid', 'is'])
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None

find_unused_parameters = True
cudnn_benchmark = True

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
