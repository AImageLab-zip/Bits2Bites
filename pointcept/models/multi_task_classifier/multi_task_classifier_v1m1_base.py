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

    # ----------------------------------------------------------
    def forward(self, input_dict):
        point = self.backbone(input_dict)
        # global pooling (identical to DefaultClassifier)
        feat = segment_csr(point.feat, point.offset, reduce="mean")  # [B, embed_dim]

        logits = [head(feat) for head in self.heads]

        if any(k.startswith("label_") for k in input_dict.keys()):
            losses = []
            for i, logit in enumerate(logits):
                target = input_dict[f"label_{i}"].view(-1)  # each head gt
                 # Cross-Entropy
                losses.append(self.criterions[i](logit, target))
            total_loss = sum(losses)
            out = {"loss": total_loss}
            for i, l in enumerate(losses):
                out[f"loss_{i}"] = l
            return out

        # inference
        return {"logits": logits}