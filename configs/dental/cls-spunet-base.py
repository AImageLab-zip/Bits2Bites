_base_ = ["../_base_/default_runtime.py"]

## BEST 

batch_size = 8
batch_size_val = 8
epoch = 200
eval_epoch = 200
num_worker = 4
empty_cache = False
enable_amp = True
clip_grad = 1.0
fold_val = 1
run_uuid = "aaaaaa"
debug = False
wandb_run_name= "default"

dataset_type = "DentalDataset"
data_root = "data/dental_landmarks_mesh"
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
            dict(type="RandomRotate",angle=[-0.1, 0.1],axis="z",center=[0, 0, 0],p=0.5,),
            dict(type="RandomDropout", dropout_ratio=0.5, dropout_application_ratio=0.5),
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True),
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
            dict(type="GridSample", grid_size=0.01, hash_type="fnv",
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
            dict(type="GridSample", grid_size=0.01, hash_type="fnv",
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
                dict(type="GridSample", grid_size=0.01, hash_type="fnv",
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
    backbone_embed_dim=256,                 # SpUNet’s final feature size
    num_classes_list=num_classes_list,
    class_weights=[
        [0.7185, 0.7791, 3.0794],               # classe dx
        [0.7386, 0.8228, 2.3214],               # classe sx
        [0.6173, 0.6849, 1.1905, 12.5000],      # morso anteriore
        [0.4762, 16.6667, 1.1905],              # trasversale
        [1.2821, 0.8197],                       # linee mediane
    ],
    loss_type="ce",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=9,
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode=True,
    ),
)

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)

hooks = [
    # dict(type="CheckpointLoader"),
    dict(type="IterationTimer"),
    dict(type="InformationWriter"),
    dict(type="MultiClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # dict(type="PreciseEvaluator", test_last=False),       # test inference on "unseen data"
]

test = dict(type="ClsTester")       # default tester prints per-head accuracy
