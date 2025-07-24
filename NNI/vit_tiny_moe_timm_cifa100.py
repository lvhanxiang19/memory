_base_ = [
    '/data/yrguan/mmpretrain_adlith/softmoe/configs/_base_/datasets/cifar100_bs16.py',
    '/data/yrguan/mmpretrain_adlith/softmoe/configs/_base_/schedules/cifar10_bs128.py',
    '/data/yrguan/mmpretrain_adlith/softmoe/configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['softmoe'],
    allow_failed_imports=False)

model = dict(
    type='TimmClassifier',
    model_name='vit_tiny_moe_custom',  # vit_tiny_patch16_224.augreg_in21k
    with_cp=False,
    num_classes=100,
    pretrained=False,
    checkpoint_path='',
    # moe settings
    moe_blocks = {moe_blocks},
    slots_per_expert = {slots_per_expert},
    num_experts = {num_experts},
    drop_path_rate = {drop_path_rate},
    moe_droprate = {moe_droprate},
    add_noise = {add_noise},
    noise_mult = 1.0,
    # loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    loss=dict(type='CrossEntropyLoss', use_soft=True, loss_weight=1.0),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)

# load_from = './softmoe/weights/vit_tiny.pth'

# dataset settings
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    # persistent_workers=True,
    # pin_memory=True,
)

# val_dataloader = dict(
#     batch_size=512,
#     num_workers=8,
#     persistent_workers=True,
#     pin_memory=True,
# )

# schedule settings
optim_wrapper = dict(
    # type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr={lr},
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.95)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={{
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }}),
    # clip_grad=dict(max_norm=5.0, norm_type=2),  # max_norm=0.1
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
        eta_min=1e-6, 
        by_epoch=True, 
        begin=50),
]

# runtime settings
# custom_hooks = [dict(type='EMAHook', momentum=1e-4)]
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=3)

randomness = dict(seed=3407, deterministic=False)
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=True,
)
default_hooks = dict(
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=50),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=3),
)
# find_unused_parameters=True
# resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
# auto_scale_lr = dict(base_batch_size=4096)
