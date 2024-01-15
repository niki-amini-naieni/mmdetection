_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
# https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#inheritance-between-configuration-files
custom_imports=dict(
    imports=['mmdet.models.losses.loc_loss'])

data_root = 'data/fsc147_grounding_dino/'
class_names = ('alcohol bottle', 'ant', 'apple', 'baguette roll', 'ball', 'banana', 'bead', 'bee', 'bird', 'birthday candle', 'biscuit', 'boat', 'book', 'bottle', 'bottle cap', 'bowl', 'box', 'bread roll', 'brick', 'buffalo', 'bullet', 'bun', 'calamari ring', 'camel', 'can', 'candle', 'candy piece', 'cap', 'car', 'carrom board piece', 'cartridge', 'cashew nut', 'cassette', 'cement bag', 'cereal', 'chair', 'chewing gum piece', 'chicken wing', 'chopstick', 'clam', 'coffee bean', 'coin', 'comic book', 'cotton ball', 'cow', 'crab cake', 'crane', 'crayon', 'croissant', 'crow', 'cup', 'cupcake', 'cupcake holder', 'deer', 'donut', 'donut holder', 'egg', 'elephant', 'finger food', 'fish', 'flamingo', 'flower', 'flower pot', 'fresh cut', 'gemstone', 'go game', 'goat', 'goldfish snack', 'goose', 'grape', 'green pea', 'horse', 'hot air balloon', 'ice cream', 'ice cream cone', 'instant noodle', 'jade stone', 'jeans', 'keyboard key', 'kidney bean', 'kitchen towel', 'kiwi', 'lego', 'lighter', 'lipstick', 'm&m piece', 'macaron', 'marble', 'marker', 'match', 'meat skewer', 'milk carton', 'mini blind', 'mosaic tile', 'naan bread', 'nail', 'nail polish', 'nut', 'onion ring', 'orange', 'oyster', 'oyster shell', 'peach', 'pearl', 'pen', 'pencil', 'penguin', 'pepper', 'person', 'pigeon', 'pill', 'plate', 'polka dot', 'polka dot tile', 'potato', 'potato chip', 'prawn cracker', 'red bean', 'rice bag', 'roof tile', 'round dessert', 'sauce bottle', 'sausage', 'screw', 'sea shell', 'seagull', 'shallot', 'sheep', 'shirt', 'shoe', 'skateboard', 'ski', 'spoon', 'spring roll', 'stair', 'stamp', 'stapler pin', 'sticky note', 'straw', 'strawberry', 'sunglass', 'supermarket shelf', 'swan', 'toilet paper roll', 'tomato', 'tree log', 'watch', 'watermelon', 'window', 'zebra')
num_classes = len(class_names)
metainfo = dict(classes=class_names)

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
                ]))
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/')))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test.json')

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
