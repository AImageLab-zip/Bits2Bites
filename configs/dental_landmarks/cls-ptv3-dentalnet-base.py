_base_ = ["../_base_/default_runtime.py"]

batch_size = 128            # to adjust
batch_size_val = 128
epoch = 100
eval_epoch = 100
empty_cache = False
enable_amp = True

dataset_type = "DentalDataset"
data_root = "data/dental_landmarks"
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
        split="val",
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
        test_cfg=dict(
            post_transform=[
                dict(type="GridSample", grid_size=0.008, hash_type="fnv",
                     mode="train", return_grid_coord=True),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord"),
                    feat_keys=["coord", "point_label_onehot"]
                )
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[1, 1], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.95, 1.05], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.95, 1.05], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.95, 1.05], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.95, 1.05], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.95, 1.05], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.95, 1.05], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.95, 1.05], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.95, 1.05], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.95, 1.05], anisotropic=True)]
            ]
        )
    ),
)

# Model
model = dict(
    type="MultiTaskClassifier",
    embed_dim=512,                          # Point-Transformer v3 default
    num_classes_list=num_classes_list,
    loss_type="ce",
    backbone=dict(                          #  Backbone: Point Transformer V3 (v1m1) and multi-task dental data
        type="PT-v3m1",                     # as in cls-ptv3-v1m1-0-base
        in_channels=9,                      # 3 xyz + 6 one-hot features
        enable_flash=False,                 # True if AMPERE gpu arch
        cls_mode=True,
    ),
)

optimizer = dict(type="AdamW", lr=6e-4, weight_decay=0.01)
scheduler = dict(type="CosineAnnealingLR", total_steps=epoch)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer"),
    dict(type="InformationWriter"),
    dict(type="MultiClsEvaluator"),
    # dict(type="PreciseEvaluator", test_last=False),       # test inference on "unseen data"
]

test = dict(type="ClsTester")       # default tester prints per-head accuracy
