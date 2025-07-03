import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import segment_csr
from ..builder import MODELS
from ..builder import build_model
from ..losses.misc import CrossEntropyLoss

@MODELS.register_module()
class MultiTaskClassifier(nn.Module):
    """
    Backbone + *k* independent classifier heads.
    """

    def __init__(
        self,
        backbone,
        num_classes_list,
        embed_dim,
        loss_type="ce",
        class_weights=None,
    ):
        super().__init__()

        self.backbone = build_model(backbone)
        self.num_tasks = len(num_classes_list)
        self.num_classes_list = list(num_classes_list)

        # one MLP head per task  (Linear-BN-ReLU-Drop-Linear)
        self.heads = nn.ModuleList()
        for n_cls in num_classes_list:
            self.heads.append(
                nn.Sequential(
                    nn.Linear(embed_dim, 256, bias=False),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, n_cls, bias=True),
                )
            )
        # CrossEntropy
        self.criterions = nn.ModuleList()
        for i, n_cls in enumerate(num_classes_list):
            w = None
            if class_weights and class_weights[i] is not None:
                w = torch.tensor(class_weights[i], dtype=torch.float32)
            self.criterions.append(CrossEntropyLoss(weight=w, ignore_index=-1))

    def forward(self, input_dict):
        point = self.backbone(input_dict)

        # pooling globale
        feat = segment_csr(point.feat, point.offset, reduce="mean")  # [B, embed_dim]

        # logit per ogni task
        logits = [head(feat) for head in self.heads]

        # creo una sola volta cls_logits e poi lo ri-uso
        cls_logits = torch.cat(logits, dim=1)        # [B, sum(num_classes_list)]

        if any(k.startswith("label_") for k in input_dict):
            losses = []
            for i, (logit, n_cls) in enumerate(zip(logits, self.num_classes_list)):
                target = input_dict[f"label_{i}"].view(-1)

                # --- Sanity check (will raise *before* we touch CUDA kernels)
                if (target < 0).any() or (target >= n_cls).any():
                    bad_vals = target[(target < 0) | (target >= n_cls)].unique()
                    raise ValueError(
                        f"Head {i}: found invalid label(s) {bad_vals.tolist()} "
                        f"for n_classes = {n_cls}. Valid range is [0, {n_cls-1}]."
                    )

                if target.shape[0] != feat.shape[0]:
                    print(f"[WARN] Adjusting label_{i} from {target.shape[0]} to {feat.shape[0]}")
                    target = target[:feat.shape[0]]

                loss_val = self.criterions[i](logit, target)
                if loss_val.ndim > 0:
                    loss_val = loss_val.mean()
                losses.append(loss_val)

            total_loss = sum(losses)
            out = {"loss": total_loss, "cls_logits": cls_logits}
            for i, l in enumerate(losses):
                out[f"loss_{i}"] = l
            return out

        # inference (niente label)
        return {"logits": logits, "cls_logits": cls_logits}