_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
# https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#inheritance-between-configuration-files
custom_imports=dict(
    imports=['mmdet.models.losses.loc_loss', 'mmdet.evaluation.metrics.count_metric'])
#import tempfile
#tempdir = tempfile.gettempdir()

data_root = 'data/fsc147_grounding_dino/'
train_class_names = ('alcohol bottle', 'baguette roll', 'ball', 'banana', 'bead', 'bee', 'birthday candle', 'biscuit', 'boat', 'bottle', 'bowl', 'box', 'bread roll', 'brick', 'buffalo', 'bun', 'calamari ring', 'can', 'candle', 'cap', 'car', 'cartridge', 'cassette', 'cement bag', 'cereal', 'chewing gum piece', 'chopstick', 'clam', 'coffee bean', 'coin', 'cotton ball', 'cow', 'crane', 'crayon', 'croissant', 'crow', 'cup', 'cupcake', 'cupcake holder', 'fish', 'gemstone', 'go game piece', 'goat', 'goldfish snack', 'goose', 'ice cream', 'ice cream cone', 'instant noodle', 'jade stone', 'jeans', 'kidney bean', 'kitchen towel', 'lighter', 'lipstick', 'm&m piece', 'macaron', 'match', 'meat skewer', 'mini blind', 'mosaic tile', 'naan bread', 'nail', 'nut', 'onion ring', 'orange', 'pearl', 'pen', 'pencil', 'penguin', 'pepper', 'person', 'pigeon', 'plate', 'polka dot tile', 'potato', 'rice bag', 'roof tile', 'screw', 'shoe', 'spoon', 'spring roll', 'stair', 'stapler pin', 'straw', 'supermarket shelf', 'swan', 'tomato', 'watermelon', 'window', 'zebra')
num_train_classes = len(train_class_names)
train_metainfo = dict(classes=train_class_names)

val_class_names = ('ant', 'bird', 'book', 'bottle cap', 'bullet', 'camel', 'chair', 'chicken wing', 'donut', 'donut holder', 'flamingo', 'flower', 'flower pot', 'grape', 'horse', 'kiwi', 'milk carton', 'oyster', 'oyster shell', 'package of fresh cut fruit', 'peach', 'pill', 'polka dot', 'prawn cracker', 'sausage', 'seagull', 'shallot', 'shirt', 'skateboard', 'toilet paper roll')
num_val_classes = len(val_class_names)
val_metainfo = dict(classes=val_class_names)

test_class_names = ('apple', 'candy piece', 'carrom board piece', 'cashew nut', 'comic book', 'crab cake', 'deer', 'egg', 'elephant', 'finger food', 'green pea', 'hot air balloon', 'keyboard key', 'lego', 'marble', 'marker', 'nail polish', 'potato chip', 'red bean', 'round dessert', 'sauce bottle', 'sea shell', 'sheep', 'ski', 'stamp', 'sticky note', 'strawberry', 'sunglasses', 'tree log', 'watch')
num_test_classes = len(test_class_names)
test_metainfo = dict(classes=test_class_names)

model = dict(bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=num_train_classes,
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
        metainfo=train_metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=val_metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/')))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=test_metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

val_evaluator = dict(type='CountMetric', ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(type='CountMetric', ann_file=data_root + 'annotations/test.json')

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0),
            'language_model': dict(lr_mult=0),
        }))
#optim_wrapper = dict(optimizer=dict(lr=0.00005))