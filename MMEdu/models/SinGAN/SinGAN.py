dataset_type = 'SinGANDataset'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=False,
    train=dict(
        type=dataset_type,
        img_path='./data/singan/balloons.png',  # need to set
        min_size=25,
        max_size=250,
        scale_factor_init=0.75))

num_scales = 8  # start from zero

model = dict(
    type='SinGAN',
    generator=dict(
        type='SinGANMultiScaleGenerator',
        in_channels=3,
        out_channels=3,
        num_scales=num_scales,  # need to be specified
    ),
    discriminator=dict(
        type='SinGANMultiScaleDiscriminator',
        in_channels=3,
        num_scales=num_scales,  # need to be specified
    ),
    gan_loss=dict(type='GANLoss', gan_type='wgan', loss_weight=1),
    disc_auxiliary_loss=[
        dict(
            type='GradientPenaltyLoss',
            loss_weight=0.1,
            norm_mode='pixel',
            data_info=dict(
                discriminator='disc_partial',
                real_data='real_imgs',
                fake_data='fake_imgs'))
    ],
    gen_auxiliary_loss=dict(
        type='MSELoss',
        loss_weight=10,
        data_info=dict(pred='recon_imgs', target='real_imgs'),
    ))

train_cfg = dict(    
    use_ema = True,
    noise_weight_init=0.1,
    iters_per_scale=2000,
    curr_scale=-1,
    disc_steps=3,
    generator_steps=3,
    lr_d=0.0005,
    lr_g=0.0005,
    lr_scheduler_args=dict(milestones=[1600], gamma=0.1))

# test_cfg = None
test_cfg = dict(
    _delete_ = True,
    pkl_data = '1/models/SinGAN/singan_balloons_20210406_191047-8fcd94cf.pkl'
)

optimizer = None
lr_config = None
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=3)

custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='visual',
        interval=500,
        bgr2rgb=True,
        res_name_list=['fake_imgs', 'recon_imgs', 'real_imgs']),
    dict(
        type='PickleDataHook',
        output_dir='pickle',
        interval=-1,
        after_run=True,
        data_name_list=['noise_weights', 'fixed_noises', 'curr_stage'])
]

total_iters = 18000


# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=True,
    pass_training_status=True)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 10000)]
find_unused_parameters = True
cudnn_benchmark = True

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
