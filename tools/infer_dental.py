from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.engines.launch import launch
from pointcept.datasets import collate_fn
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F

num_classes_list = [3, 3, 4, 3, 2]  # define this near top of your script

def main_worker(cfg):
    cfg = cfg[0] if isinstance(cfg, tuple) else cfg
    cfg = default_setup(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset: use test set, but no TTA
    dataset = build_dataset(cfg.data.test)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get("num_worker", 4),
        pin_memory=True,
        collate_fn=collate_fn  # ← usa lo stesso collate_fn del test originale
    )

    # Model
    model = build_model(cfg.model)
    model.load_state_dict(torch.load(cfg.weight, map_location=device)["state_dict"])
    model = model.to(device)
    model.eval()

    # Inference loop
    print(f"Inferenza su {len(dataset)} campioni…")
    for i, batch in enumerate(tqdm(dataloader)):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        with torch.no_grad():
            output = model(batch)

        logits = output["logits"]
        assert len(logits) == len(num_classes_list), "Mismatch between logits and num_classes_list"

        name = dataset.get_data_name(i)
        print(f"\n→ {name}:")

        for j, logit in enumerate(logits):
            pred = torch.argmax(logit, dim=1).item()
            probs = F.softmax(logit, dim=1).squeeze().tolist()
            print(f"  label_{j} (classes: {num_classes_list[j]}):")
            print(f"    → Predicted class: {pred}")
            print(f"    → Probabilities: {[f'{p:.2f}' for p in probs]}")

def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    # Richiede opzione `--options weight=...` oppure set manuale
    assert cfg.get("weight", None) is not None, "Passa il peso con --options weight=path_to_model.pth"

    cfg.save_path = os.path.join("infer_output", os.path.basename(cfg.weight).replace(".pth", ""))
    os.makedirs(cfg.save_path, exist_ok=True)

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
