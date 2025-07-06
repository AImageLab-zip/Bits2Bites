_base_ = ["../_base_/default_runtime.py"]

batch_size = 16            # adjust
epoch = 250
empty_cache = False
enable_amp = True

dataset_type = "DentalDataset"
data_root = "data/dental"
num_classes_list = [3, 3, 4, 3, 2]   # class sizes

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomShift", shift=((-0.02,0.02),)*3),
            dict(type="GridSample", grid_size=0.008, hash_type="fnv",
                 mode="train", return_grid_coord=True),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord", "grid_coord",
                    "label_0", "label_1", "label_2", "label_3", "label_4"),
                feat_keys=["coord", "point_label_onehot"]   # (3+6)-D features
            )
        ],
    ),
    val=dict(
        type=dataset_type,
        split="val",       # or "test" if you have only train/test
        data_root=data_root,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="GridSample", grid_size=0.008, hash_type="fnv",
                 mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord",
                      "label_0", "label_1", "label_2", "label_3", "label_4"),
                feat_keys=["coord", "point_label_onehot"]
            )
        ],
        test_mode=False,
    ),
    # inference
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="GridSample", grid_size=0.008, hash_type="fnv",
                 mode="test", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord"),
                feat_keys=["coord", "point_label_onehot"]
            )
        ],
        test_mode=True,
    ),
)

model = dict(
    type="MultiTaskClassifier",
    embed_dim=256,                 # SpUNet’s final feature size
    num_classes_list=num_classes_list,
    loss_type="ce",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=9,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode=True,
    ),
)

optimizer = dict(type="SGD", lr=0.1, momentum=0.9,
                 weight_decay=1e-4, nesterov=True)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer"),
    dict(type="InformationWriter"),
    dict(type="BestCkptSaver", metric="loss/val"),   # restore best weight ?
    dict(type="ClsEvaluator"),
    dict(type="PreciseEvaluator", test_last=False),
]

test = dict(type="ClsTester")       # default tester prints per-head accuracy
