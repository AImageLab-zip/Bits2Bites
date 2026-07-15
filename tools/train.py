"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
import os
from dental_fold import prepare_folds

def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    cfg.wandb_project = os.environ.get("WANDB_PROJECT", "default_project")
    cfg.wandb_key = os.environ.get("WANDB_KEY", "no_key")
    cfg.wandb_entity = os.environ.get("WANDB_ENTITY", "default_entity")
    cfg.wandb_run_name = os.environ.get("WANDB_RUN_NAME", "default")

    cfg.run_uuid = os.environ.get("PROJECT_UUID", "aaaaaa")
    cfg.dataset_type=dict(
        type='DentalDataset',
        data_root=os.path.join("data", "dt_" + cfg.run_uuid),
        debug=cfg.debug
    )
    cfg.save_path = os.path.join("exp", "dt_" + cfg.run_uuid)
    cfg.data_root = os.path.join("data", "dt_" + cfg.run_uuid)
    prepare_folds(cfg.fold_val, cfg.data_root)

    cfg.data.train.data_root = cfg.data_root
    cfg.data.val.data_root = cfg.data_root
    cfg.data.test.data_root = cfg.data_root

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
