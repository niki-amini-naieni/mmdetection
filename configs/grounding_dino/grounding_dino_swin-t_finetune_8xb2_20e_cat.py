_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
custom_imports=dict(
    imports=['mmdet.models.losses.loc_loss', 'mmdet.evaluation.metrics.count_metric'])

data_root = 'data/cat/'
class_name = ('cat',)
num_classes = len(class_name)
#num_classes = 1e6
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale=0.0, bias=False),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='LocLoss', loss_weight=1.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0)
    ),
    train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='BinaryFocalLossCost', weight=5.0),
                    dict(type='LocCost', weight=1.0, box_format='xywh')
                ])),
    test_cfg=dict(max_per_img=900)
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/trainval.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

val_evaluator = dict(type='CountMetric', ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)
