from abc import abstractmethod

import torch
from torch import Tensor, nn

from chemprop.nn.hparams import HasHParams
from chemprop.utils import ClassRegistry

__all__ = [
    "Aggregation",
    "AggregationRegistry",
    "MeanAggregation",
    "SumAggregation",
    "NormAggregation",
    "AttentiveAggregation",
]


class Aggregation(nn.Module, HasHParams):
    """An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
    a batch of graph-level representations

    .. note::
        this class is abstract and cannot be instantiated.

    See also
    --------
    :class:`~chemprop.v2.models.modules.agg.MeanAggregation`
    :class:`~chemprop.v2.models.modules.agg.SumAggregation`
    :class:`~chemprop.v2.models.modules.agg.NormAggregation`
    """

    def __init__(self, dim: int = 0, *args, **kwargs):
        super().__init__()
        
        self.dim = dim
        self.hparams = {"dim": dim, "cls": self.__class__}

    @abstractmethod
    def forward(self, H: Tensor, batch: Tensor, fragment_mask: Tensor) -> Tensor:
        """Aggregate only real atoms (exclude fragments)
        
        Parameters
        ----------
        H : Tensor
            Node representations (V x d)
        batch : Tensor
            Batch indices for each node (V)
        fragment_mask : Tensor
            Boolean mask where True indicates fragment nodes (V)
            
        Returns
        -------
        Tensor
            Graph-level representations (b x d)
        """
        

AggregationRegistry = ClassRegistry[Aggregation]()



@AggregationRegistry.register("mean")
class MeanAggregation(Aggregation):
    def forward(self, H: Tensor, batch: Tensor, fragment_mask: Tensor) -> Tensor:
        # Filter out fragment nodes
        real_atom_mask = ~fragment_mask
        H_real = H[real_atom_mask]
        batch_real = batch[real_atom_mask]
         
        index_torch = batch_real.unsqueeze(1).repeat(1, H_real.shape[1])
        dim_size = batch_real.max().int() + 1
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, H_real, reduce="mean", include_self=False
        )


@AggregationRegistry.register("sum")
class SumAggregation(Aggregation):
    def forward(self, H: Tensor, batch: Tensor, fragment_mask: Tensor) -> Tensor:
        # Filter out fragment nodes
        real_atom_mask = ~fragment_mask
        H_real = H[real_atom_mask]
        batch_real = batch[real_atom_mask]
        
        if len(batch_real) == 0:
            return torch.zeros(batch.max()+1, H.shape[1], dtype=H.dtype, device=H.device)
            
        index_torch = batch_real.unsqueeze(1).repeat(1, H_real.shape[1])
        dim_size = batch_real.max().int() + 1
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, H_real, reduce="sum", include_self=False
        )


@AggregationRegistry.register("norm")
class NormAggregation(SumAggregation):
    def __init__(self, dim: int = 0, *args, norm: float = 100.0, **kwargs):
        super().__init__(dim, **kwargs)
        self.norm = norm
        self.hparams["norm"] = norm

    def forward(self, H: Tensor, batch: Tensor, fragment_mask: Tensor) -> Tensor:
        return super().forward(H, batch, fragment_mask) / self.norm


class AttentiveAggregation(Aggregation):
    def __init__(self, dim: int = 0, *args, output_size: int, **kwargs):
        super().__init__(dim, *args, **kwargs)
        self.hparams["output_size"] = output_size
        self.W = nn.Linear(output_size, 1)

    def forward(self, H: Tensor, batch: Tensor, fragment_mask: Tensor) -> Tensor:
        # Filter out fragment nodes
        real_atom_mask = ~fragment_mask
        H_real = H[real_atom_mask]
        batch_real = batch[real_atom_mask]
        
        if len(batch_real) == 0:
            return torch.zeros(batch.max()+1, H.shape[1], dtype=H.dtype, device=H.device)
            
        dim_size = batch_real.max().int() + 1
        attention_logits = self.W(H_real).exp()
        
        Z = torch.zeros(dim_size, 1, dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, batch_real.unsqueeze(1), attention_logits, reduce="sum", include_self=False
        )
        
        alphas = attention_logits / Z[batch_real]
        index_torch = batch_real.unsqueeze(1).repeat(1, H_real.shape[1])
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, alphas * H_real, reduce="sum", include_self=False
        )