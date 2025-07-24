_base_ = [
    '/home/xkzhu/workspace/mmpretrain/mmpretrain/softmoe/configs/_base_/datasets/DTD_bs1.py',
    '/home/xkzhu/workspace/mmpretrain/mmpretrain/softmoe/configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['softmoe'],
    allow_failed_imports=False)

# hyper_parameters
n_GPU = 1
warmup_epochs = 100
lr = {lr}
bs = 512
accumulative_counts = 512//(n_GPU*bs)

dp = {dp}
moe_dp = {moe_dp}
moe_dr = {moe_dr}
c_EXPs = 98
u_EXPs = 196
univ_factor = 1/4
n_slots = 1
layer_decay_rate = {layer_decay_rate}

moe_groups_dict = {{}}
# phi_groups_dict = None
phi_groups_dict = {{}}
moe_layers = [6,7,8,9,10,11]
load_from = '/home/xkzhu/workspace/mmpretrain/mmpretrain/softmoe/weights/gen_weight/Dual-MoE-L2-mlp_[D_vit_t]+[E_vit_s+vit_t_univ]_selected.pt'
moe_mult = 1.0

model = dict(
    type='TimmClassifier',
    model_name='vit_tiny_moe_custom',  # vit_tiny_patch16_224.augreg_in21k
    with_cp=True,
    num_classes=47,
    pretrained=False,
    checkpoint_path='',
    global_pool = 'avg',  # token, avg
    # moe settings
    only_phi = False,
    moe_groups_dict = moe_groups_dict,
    phi_groups_dict = phi_groups_dict,
    moe_layers=moe_layers,
    slots_per_expert = n_slots,
    num_experts = 197,
    # dualpath moe
    if_dualpath = True,
    core_experts = c_EXPs,
    univ_experts = u_EXPs,
    univ_factor = univ_factor,
    layer_scale=False,
    moe_droprate = moe_dr,
    moe_drop_path_rate = moe_dp,
    drop_path_rate = dp,
    add_noise = True,
    noise_mult = 1/(c_EXPs+u_EXPs),
    compress_ratio = 1.0,
    loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    # loss=dict(type='CrossEntropyLoss', use_soft=True, loss_weight=1.0),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)


# dataset settings
train_dataloader = dict(
    batch_size=bs,
    num_workers=12,
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
DECAY_MULT = 0.0
def generate_moe_custom_keys(layer_keys, lr_mult=1.0, decay_mult=0.0):
    keys = {{}}
    lr_mult_param = ['mlp.phi', 'mlp.scale',
                     'mlp.experts.vit_fc1.weight', 'mlp.experts.vit_fc2.weight']
    decay_mult_param = []

    for key in layer_keys:
        for param in lr_mult_param:
            keys[f'blocks.{{key}}.{{param}}'] = {{'lr_mult': lr_mult}}
        for param in decay_mult_param:
            keys[f'blocks.{{key}}.{{param}}'] = {{'decay_mult': decay_mult}}
    
    return keys

custom_keys = generate_moe_custom_keys(moe_layers, lr_mult=moe_mult, decay_mult=DECAY_MULT)
common_keys = {{
    '.cls_token': dict(decay_mult=DECAY_MULT),
    '.pos_embed': dict(decay_mult=DECAY_MULT),
}}
custom_keys.update(common_keys)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    # constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=layer_decay_rate,
        norm_decay_mult=DECAY_MULT,
        bias_decay_mult=DECAY_MULT,
        custom_keys=custom_keys,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2),  # max_norm=0.1
    accumulative_counts=accumulative_counts,
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR', 
        eta_min_ratio=1e-2,
        by_epoch=True, 
        begin=warmup_epochs),
]

# runtime settings
# custom_hooks = [dict(type='EMAHook', momentum=1e-4)]

train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=6)
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
    checkpoint=dict(type='CheckpointHook', interval=30, max_keep_ckpts=1),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
# auto_scale_lr = dict(base_batch_size=4096)
