from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Sequence, Callable

import numpy as np
import torch
from torch import Tensor

from chemprop.data.datasets import Datum
from chemprop.data.molgraph import MolGraph

from rdkit import Chem
from rdkit.Chem import BRICS
from chemprop.featurizers import V1RDKit2DNormalizedFeaturizer

@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`\s.

    It has all the attributes of a ``MolGraph`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data
    """

    mgs_input: InitVar[Sequence[MolGraph]]  
    """A list of individual :class:`MolGraph`\s to be batched together"""
    V: Tensor = field(init=False)
    """the atom feature matrix"""
    E: Tensor = field(init=False)
    """the bond feature matrix"""
    edge_index: Tensor = field(init=False)
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor = field(init=False)
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    batch: Tensor = field(init=False)
    """the index of the parent :class:`MolGraph` in the batched graph"""
    fragment_mask: Tensor =field(init=False)
    mgs: Sequence[MolGraph] =field(init=False) 

    __size: int = field(init=False)

    def __post_init__(self, mgs_input: Sequence[MolGraph]):
        self.__size = len(mgs_input)

        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs_input):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))   
            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()
        self.mgs=mgs_input
        

    def fragmentation(self, mgs : Sequence[MolGraph] ,linear: Callable[[torch.Tensor], torch.Tensor]):
        """
        Initialize a BatchMolGraph from a list of MolGraph objects, enriching each one 
        with fragment-based features.
        
        Args:
            mgs: A list of individual MolGraphs to be batched together
        """
        
        # Process each MolGraph to add fragment information
        enriched_mgs = [self._enrich_with_fragments(mg,linear) for mg in mgs]
        
        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []
        
        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(enriched_mgs):
            Vs.append(mg[0])  # Enhanced V with fragment nodes
            Es.append(mg[1])  # Enhanced E with fragment edges
            edge_indexes.append(mg[2] + num_nodes)  # Enhanced edge_index
            rev_edge_indexes.append(mg[3] + num_edges)  # Enhanced rev_edge_index
            batch_indexes.append([i] * mg[0].shape[0])  # Batch indices for all nodes including fragments
            
            num_nodes += mg[0].shape[0]
            num_edges += mg[2].shape[1]
        
        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()
        
        # Store fragment mask to identify fragment nodes vs. atom nodes
        self.fragment_mask = torch.zeros(self.V.shape[0], dtype=torch.bool)
        offset = 0
        for mg in enriched_mgs:
            fragment_mask = np.zeros(mg[0].shape[0], dtype=bool)
            if len(mg) > 4:  # If fragment info exists
                fragment_mask[mg[4]:] = True  # Mark fragment nodes
            self.fragment_mask[offset:offset+mg[0].shape[0]] = torch.tensor(fragment_mask)
            offset += mg[0].shape[0]

    
    def _enrich_with_fragments(self, mg: MolGraph,linear: Callable[[torch.Tensor], torch.Tensor]) -> tuple:
        """
        Enrich a MolGraph with fragment-based features by decomposing the molecule
        and adding fragment descriptor nodes to the graph.
        
        Args:
            mg: Original MolGraph
            
        Returns:
            Tuple containing enriched (V, E, edge_index, rev_edge_index, num_atoms)
        """
        # Extract original components
        V_orig = mg.V
        E_orig = mg.E
        edge_index_orig = mg.edge_index
        rev_edge_index_orig = mg.rev_edge_index
        mol = mg.mol
        
        if mol is None:
            # If no molecule is available, return the original graph unchanged
            return (V_orig, E_orig, edge_index_orig, rev_edge_index_orig)
        
        # Get feature dimensions
        atom_feat_dim = V_orig.shape[1]
        bond_feat_dim = E_orig.shape[1]
        num_atoms = V_orig.shape[0]
        
        # Get BRICS fragments and atom-to-fragment mapping
        fragments, atom_to_fragment = self._get_brics_fragments(mol)
        num_fragments = len(fragments)
        
        if num_fragments == 0:
            # If no fragments are found, return the original graph unchanged
            return (V_orig, E_orig, edge_index_orig, rev_edge_index_orig)
        
        # Compute fragment descriptors with the same dimension as atom features
        fragment_features = np.vstack([
            self._compute_fragment_descriptors(frag, linear) 
            for frag in fragments
        ])
        
        # Create new combined node features (atoms + fragment descriptors)
        V_combined = np.vstack([V_orig, fragment_features])
        
        # Create edges between fragment nodes and their atoms
        new_edges = []
        for atom_idx, frag_idx in atom_to_fragment.items():
            # Create edges in both directions (atom to fragment and fragment to atom)
            fragment_node_idx = num_atoms + frag_idx
            new_edges.append([atom_idx, fragment_node_idx])
            new_edges.append([fragment_node_idx, atom_idx])
        
        new_edges = np.array(new_edges).T if new_edges else np.zeros((2, 0), dtype=int)
        
        # Create features for new edges (initialized to zeros as specified)
        num_new_edges = new_edges.shape[1]
        E_new = np.zeros((num_new_edges, bond_feat_dim), dtype=np.float32)
        
        # Combine original and new edges
        edge_index_combined = np.hstack([edge_index_orig, new_edges])
        
        # Update reverse edge index mapping
        num_orig_edges = edge_index_orig.shape[1]
        rev_edge_index_new = np.arange(num_orig_edges, num_orig_edges + num_new_edges)
        # For each new edge, the reverse is the next one (or previous one for odd indices)
        for i in range(0, num_new_edges, 2):
            if i + 1 < num_new_edges:
                rev_edge_index_new[i] = num_orig_edges + i + 1
                rev_edge_index_new[i + 1] = num_orig_edges + i
        
        rev_edge_index_combined = np.concatenate([rev_edge_index_orig, rev_edge_index_new])
        
        # Combine original and new edge features
        E_combined = np.vstack([E_orig, E_new]) if num_new_edges > 0 else E_orig
        
        return (V_combined, E_combined, edge_index_combined, rev_edge_index_combined, num_atoms)
    
    def _get_brics_fragments(self, mol: Chem.Mol) -> tuple[list[Chem.Mol], dict[int, int]]:
        """
        Decompose a molecule using BRICS and return fragments and atom-to-fragment mapping.
        
        Args:
            mol: RDKit molecule to decompose
        
        Returns:
            fragments: List of RDKit molecules representing fragments
            atom_to_fragment: Dictionary mapping atom indices to fragment indices
        """
        Chem.Kekulize(mol, clearAromaticFlags=True)
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        if not brics_bonds:  # If no BRICS bonds are found, treat the whole molecule as one fragment
            atom_to_fragment = {i: 0 for i in range(mol.GetNumAtoms())}
            return [mol], atom_to_fragment
        
        # Break molecule at BRICS bonds
        break_bonds = [(bond[0][0], bond[0][1]) for bond in brics_bonds]
        
        # Create a copy to work with
        mol_copy = Chem.Mol(mol)
        
        # Map atoms in the molecule to keep track of them
        for atom in mol_copy.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        
        # Remove bonds at break points
        rwmol = Chem.RWMol(mol_copy)
        for a1, a2 in break_bonds:
            rwmol.RemoveBond(a1, a2)
        
        # Get fragments as connected components
        fragments = Chem.GetMolFrags(rwmol, asMols=True, sanitizeFrags=True)
        
        # Map original atom indices to fragment indices
        atom_to_fragment = {}
        for frag_idx, frag in enumerate(fragments):
            # Get mapping from fragment atoms back to original molecule
            for atom in frag.GetAtoms():
                atom_map_num = atom.GetAtomMapNum()
                if atom_map_num > 0:
                    atom_to_fragment[atom_map_num - 1] = frag_idx
        
        return fragments, atom_to_fragment
    
    def _compute_fragment_descriptors(self, fragment: Chem.Mol, linear: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        """
        Compute descriptor features for a molecular fragment.
        
        Args:
            fragment: RDKit molecule representing a fragment
            linear: fucntion to reduce the dim of the descriptors
        
        Returns:
            Descriptor vector for the fragment with the specified dimension
        """
        generator = V1RDKit2DNormalizedFeaturizer()
        fp = generator(fragment)  # This is a NumPy array
        fp_tensor = torch.from_numpy(fp).float()  # Convert to torch.Tensor
        
        # ✅ Move to same device as linear's parameters
        device = next(linear.parameters()).device
        fp_tensor = fp_tensor.to(device)
        
        features = linear(fp_tensor)  # Now safe to pass to linear
        return features.detach().cpu().numpy()



    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: torch.device | str):
        """Moves all tensor attributes to the specified device."""
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)
        if hasattr(self, "fragment_mask"):
            self.fragment_mask = self.fragment_mask.to(device)


class TrainingBatch(NamedTuple):
    bmg: BatchMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return TrainingBatch(
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


class MulticomponentTrainingBatch(NamedTuple):
    bmgs: list[BatchMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_multicomponent(batches: Iterable[Iterable[Datum]]) -> MulticomponentTrainingBatch:
    tbs = [collate_batch(batch) for batch in zip(*batches)]

    return MulticomponentTrainingBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )
