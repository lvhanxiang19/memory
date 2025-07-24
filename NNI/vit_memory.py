_base_ = [
   #'../_base_/datasets/cifar100_bs16.py',
    '/home/ipad_3d/jet/jet/moejet/configs/_base_/default_runtime.py',
    #'/home/ipad_3d/jet/jet/moejet/configs/_base_/schedules/cifar10_bs128.py'
]


custom_imports = dict(
    imports=['moejet'],
    allow_failed_imports=False)

# hyper_parameters
n_GPU = 4
bs = 512
bs_gpu=int(bs/n_GPU)


#data_setting
cifar10_pre=dict(
    num_classes=10,
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    # loaded images are already RGB format
    to_rgb=False)
cifar100_pre=dict(
    num_classes=100,
    # RGB format normalization parameters
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    # loaded images are already RGB format
    to_rgb=False)
dataset_type = 'CIFAR100'#dataset_type = 'CIFAR10'
data_preprocessor=cifar100_pre if dataset_type=='CIFAR100' else cifar10_pre 
num_class=100 if dataset_type=='CIFAR100' else 10
bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=bs_gpu,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/cifar100',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=bs_gpu,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/cifar100/',
        split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator



#model setting
model = dict(
    type='TimmClassifier',
    model_name='vit_tiny_mem_custom',  # vit_tiny_patch16_224.augreg_in21k
    with_cp=False,
    num_classes=num_class,
    pretrained=False,
     #memory args
    mem_dim=192,
    mem_k_dim=192,
    mem_k_num={k_num},
    mem_head={head},
    mem_layer={memory_layer},
    mem_knn={knn},
    value_lr={value_lr},
    gate={gate},
    #
    loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    # loss=dict(type='CrossEntropyLoss', use_soft=True, loss_weight=1.0),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ])
)
DECAY_MULT = 0.0
layer_decay_rate = 0.6
optim_wrapper = dict(
    #type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=4e-3,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    #constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        #layer_decay_rate=layer_decay_rate,
        norm_decay_mult=DECAY_MULT,
        bias_decay_mult=DECAY_MULT,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2),  # max_norm=0.1
    accumulative_counts=1,
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=50,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR', 
        T_max=300,
        eta_min_ratio=1e-2,
        by_epoch=True, 
        begin=50),
]
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=3)
val_cfg = dict()
test_cfg = dict()

randomness = dict(seed=3407, deterministic=False)
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=True,
)
default_hooks = dict(
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=3, max_keep_ckpts=1),
)

# schedules settings

# train, val, test setting
