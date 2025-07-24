_base_ = [
    '../_base_/datasets/cifar100_bs16.py',
    '../_base_/default_runtime.py',
]
from memory.models import timm_model_register
custom_imports = dict(
    imports=['memory'],
    allow_failed_imports=False,)
n_GPU = 8

is_new=False
mem_k_num=256
is_peer=False
layers=[1,3,5,7,9,11]
total_batch = 512
lr = 4e-3
bs = 128
h=4

accumulative_counts = 1
model = dict(
    type='TimmClassifier',

    model_name='vit_tiny_mem_custom',  # vit_tiny_patch16_224.augreg_in21k
    with_cp=False,
    num_classes=100,
    pretrained=False,
    #memory args

    mem_dim=192,
    mem_k_dim=192,
    mem_k_num=mem_k_num,
    mem_head=h,
    mem_layer=layers,
    mem_knn=32,
    is_peer=is_peer,
    
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
        lr=lr,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    #constructor='LearningRateDecayOptimWrapperConstructor',

    paramwise_cfg=dict(
        #layer_decay_rate=layer_decay_rate,
        custom_keys={
                     
                     'blocks.5.mlp.weight_down_embed': dict(lr_mult=0.25, decay_mult=0.9),
                     'blocks.5.mlp.weight_up_embed': dict(lr_mult=0.25, decay_mult=0.9),
                     'blocks.5.mlp.value': dict(lr_mult=0.25, decay_mult=0.9)
                     
                     },
        norm_decay_mult=DECAY_MULT,
        bias_decay_mult=DECAY_MULT,
        bypass_duplicate=True,
    ),

    clip_grad=dict(max_norm=5.0, norm_type=2),  # max_norm=0.1
    accumulative_counts=accumulative_counts,
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
name='lhx'
# runtime settings
# custom_hooks = [dict(type='EMAHook', momentum=1e-4)]
# custom_hooks = [
#     dict(type='Fp16CompresssionHook'),
# ]

train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
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
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
)
if is_peer:
 work_dir = f'work_dirs/vit_peer_{mem_k_num}_{layers}_{h}_{lr}_new_{is_new}'
else:
 work_dir = f'work_dirs/vit_memory_{mem_k_num}_{layers}_{h}_{lr}'