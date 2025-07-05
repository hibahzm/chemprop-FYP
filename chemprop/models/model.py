from __future__ import annotations

import io
import logging
from typing import Iterable, TypeAlias

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim

from chemprop.data import BatchMolGraph, MulticomponentTrainingBatch, TrainingBatch
from chemprop.nn import Aggregation, ChempropMetric, MessagePassing, Predictor
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched
from chemprop.utils.registry import Factory

logger = logging.getLogger(__name__)

BatchType: TypeAlias = TrainingBatch | MulticomponentTrainingBatch

import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader
import torch.nn.functional as F

class ChemBERTaEncoder(nn.Module):
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MLM", fine_tune_percent=10, unfreeze_pooler=True):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.encoder = RobertaModel.from_pretrained(model_name)

        # Step 1: Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Step 2: Unfreeze the top k layers based on fine_tune_percent
        num_layers_total = len(self.encoder.encoder.layer)
        k = max(1, int(num_layers_total * fine_tune_percent / 100))

        for layer in self.encoder.encoder.layer[-k:]:  # Unfreeze top k layers
            for param in layer.parameters():
                param.requires_grad = True

        # Logging
        total = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"ChemBERTa Total parameters: {total}")
        print(f"Trainable parameters: {trainable} ({100 * trainable / total:.2f}%)")

    def encode(self, smiles_list: list[str], batch_size=64, max_length=128):
        device = next(self.encoder.parameters()).device
        all_hidden_states = []
        all_pooler_outputs = []

        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.set_grad_enabled(self.encoder.training):
                outputs = self.encoder(**inputs)
                last_hidden = outputs.last_hidden_state.detach().clone()     # [B, L, d_model]
                pooler = outputs.pooler_output.detach().clone()              # [B, d_model]
                all_hidden_states.append(last_hidden)
                all_pooler_outputs.append(pooler)

        # Return both as tensors
        return {
            "last_hidden_state": torch.cat(all_hidden_states, dim=0),
            "pooler_output": torch.cat(all_pooler_outputs, dim=0)
        }

class fusionGAT(nn.Module):
    def __init__(self, dmpnn_dim: int, bert_dim: int, hidden_dim: int):
        super().__init__()
        # Project descriptor and nodes to hidden_dim
        self.W_dmpnn = nn.Linear(dmpnn_dim, hidden_dim)
        self.W_bert = nn.Linear(bert_dim, hidden_dim)
        self.attn_fc = nn.Linear(2 * hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.W_out=nn.Linear(2*hidden_dim, hidden_dim)
    def forward(self, dmpnn_output: Tensor, encodings: Tensor) -> Tensor:
        """
        desc: (B, dmpnn_dim)
        nodes: (B, L, bert_dim)
        
        Returns:
            updated_desc: (B, hidden_dim)
        """
        B, L, _ = encodings.size()

        dmpnn_proj = self.W_dmpnn(dmpnn_output)  # (B, hidden_dim)
        bert_proj = self.W_bert(encodings)  # (B, L, hidden_dim)

        # Expand dmpnn_proj to (B, L, hidden_dim) to concatenate with each node
        dmpnn_expanded = dmpnn_proj.unsqueeze(1).expand(-1, L, -1)  # (B, L, hidden_dim)

        # Concatenate dmpnn and bert features
        cat = torch.cat([dmpnn_expanded, bert_proj], dim=-1)  # (B, L, 2*hidden_dim)

        # Compute attention scores
        e = self.leaky_relu(self.attn_fc(cat)).squeeze(-1)  # (B, L)

        # Attention weights over L nodes
        alpha = torch.softmax(e, dim=1).unsqueeze(-1)  # (B, L, 1)

        # Weighted sum of node features
        weighted = torch.sum(alpha * bert_proj, dim=1)   # (B, hidden_dim)

        concat = torch.cat([dmpnn_proj, weighted], dim=-1)  # (B, 2*hidden_dim)
        fusion = self.W_out(concat)                         # (B, hidden_dim)
        return fusion



class MPNN(pl.LightningModule):
    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
        fine_tune_bert: bool = True,
        fine_tune_percent: int = 10
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["X_d_transform", "message_passing", "agg", "predictor"])
        self.hparams["X_d_transform"] = X_d_transform
        self.hparams.update({
            "message_passing": message_passing.hparams,
            "agg": agg.hparams,
            "predictor": predictor.hparams,
        })
        
        self.fusion_GAT = fusionGAT(
            dmpnn_dim=message_passing.output_dim,
            bert_dim=768,
            hidden_dim=message_passing.output_dim
        )
        
        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.predictor = predictor
        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

        self.metrics = (
            nn.ModuleList([*metrics, self.criterion.clone()])
            if metrics
            else nn.ModuleList([self.predictor._T_default_metric(), self.criterion.clone()])
        )

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        
        self.fine_tune_bert = fine_tune_bert
        self.fine_tune_percent = fine_tune_percent
        
        
        self.bert_encoder = ChemBERTaEncoder(
            model_name="seyonec/ChemBERTa-zinc-base-v1",
            fine_tune_percent=self.fine_tune_percent if self.fine_tune_bert else 0
        )

        self.bert_encoder = self.bert_encoder.to(self.device)
        

    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self) -> ChempropMetric:
        return self.predictor.criterion

    def fingerprint(self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None) -> Tensor:
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        
        smiles_list = bmg.names
        outputs = self.bert_encoder.encode(smiles_list)
        output_pooler = outputs["last_hidden_state"]
        
        fingerprint = self.fusion_GAT(H, output_pooler)
        fingerprint = self.bn(fingerprint)
        
        return fingerprint if X_d is None else torch.cat((fingerprint, self.X_d_transform(X_d)), 1)

    def encoding(self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None, i: int = -1) -> Tensor:
        return self.predictor.encode(self.fingerprint(bmg, V_d, X_d), i)

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None) -> Tensor:
        return self.predictor(self.fingerprint(bmg, V_d, X_d))

    def training_step(self, batch: BatchType, batch_idx):
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)

        self.log("train_loss", self.criterion, batch_size=batch_size, prog_bar=True, on_epoch=True)
        return l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.message_passing.V_d_transform.train()
        self.message_passing.graph_transform.train()
        self.X_d_transform.train()
        self.predictor.output_transform.train()
        
        if self.fine_tune_bert:
            self.bert_encoder.encoder.train()

    def validation_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        self.metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)
        self.log("val_loss", self.metrics[-1], batch_size=batch_size, prog_bar=True)

    def test_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch: BatchType, label: str) -> None:
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(bmg, V_d, X_d)
        weights = torch.ones_like(weights)

        if self.predictor.n_targets > 1:
            preds = preds[..., 0]

        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f"{label}/{m.alias}", m, batch_size=batch_size)

    def predict_step(self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        bmg, X_vd, X_d, *_ = batch
        return self(bmg, X_vd, X_d)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            logger.warning(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )

        return {"optimizer": opt, "lr_scheduler": {"scheduler": lr_sched, "interval": "step"}}

    def get_batch_size(self, batch: TrainingBatch) -> int:
        return len(batch[0])

    @classmethod
    def _load(cls, path, map_location, **submodules):
        d = torch.load(path, map_location, weights_only=False)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        if hparams["metrics"] is not None:
            hparams["metrics"] = [
                cls._rebuild_metric(metric)
                if not hasattr(metric, "_defaults")
                or (not torch.cuda.is_available() and metric.device.type != "cpu")
                else metric
                for metric in hparams["metrics"]
            ]

        if hparams["predictor"]["criterion"] is not None:
            metric = hparams["predictor"]["criterion"]
            if not hasattr(metric, "_defaults") or (
                not torch.cuda.is_available() and metric.device.type != "cpu"
            ):
                hparams["predictor"]["criterion"] = cls._rebuild_metric(metric)

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        return submodules, state_dict, hparams

    @classmethod
    def _add_metric_task_weights_to_state_dict(cls, state_dict, hparams):
        if "metrics.0.task_weights" not in state_dict:
            metrics = hparams["metrics"]
            n_metrics = len(metrics) if metrics is not None else 1
            for i_metric in range(n_metrics):
                state_dict[f"metrics.{i_metric}.task_weights"] = torch.tensor([[1.0]])
            state_dict[f"metrics.{i_metric + 1}.task_weights"] = state_dict[
                "predictor.criterion.task_weights"
            ]
        return state_dict

    @classmethod
    def _rebuild_metric(cls, metric):
        return Factory.build(metric.__class__, task_weights=metric.task_weights, **metric.__dict__)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MPNN:
        submodules = {
            k: v for k, v in kwargs.items() if k in ["message_passing", "agg", "predictor"]
        }
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)
        d = torch.load(checkpoint_path, map_location, weights_only=False)
        d["state_dict"] = state_dict
        d["hyper_parameters"] = hparams
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)

        return super().load_from_checkpoint(buffer, map_location, hparams_file, strict, **kwargs)

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True, **submodules) -> MPNN:
        submodules, state_dict, hparams = cls._load(model_path, map_location, **submodules)
        hparams.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model
