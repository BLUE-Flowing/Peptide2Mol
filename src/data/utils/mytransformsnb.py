import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax
from torch_geometric.transforms import Compose  # imported by train.py
import traceback
# from models.transition import *
from src.data.utils.data import Drug3DData
# from src.data.utils.dataset import *
# from src.data.utils.misc import *


class FeaturizeMol(object):
    def __init__(self, 
                atomic_numbers=[6,7,8,9,15,16,17], 
                mol_bond_types=[1, 2, 3, 4, 5],
                 use_mask_node=True, 
                 use_mask_edge=True):
        super().__init__()
        self.atomic_numbers = torch.LongTensor(atomic_numbers)
        self.mol_bond_types = torch.LongTensor(mol_bond_types)
        self.num_element = self.atomic_numbers.size(0)
        self.num_bond_types = self.mol_bond_types.size(0)

        self.num_node_types = self.num_element + int(use_mask_node)
        self.num_edge_types = self.num_bond_types + int(use_mask_edge) # + 1 for the non-bonded edges
        self.use_mask_node = use_mask_node
        self.use_mask_edge = use_mask_edge
        
        self.ele_to_nodetype = {ele: i for i, ele in enumerate(atomic_numbers)}
        self.nodetype_to_ele = {i: ele for i, ele in enumerate(atomic_numbers)}
        
        
        self.follow_batch = ['node_type', 'halfedge_type','diff_bond_type_idx', 'diffu_idx']
        self.exclude_keys = ['orig_keys', 'pos_all_confs', 'smiles', 
                             'bond_index', 'bond_type', 'num_bonds', 'num_atoms']
    
    def __call__(self, data: Drug3DData):
        
        data.num_nodes = data.num_atoms
        data.node_type = data.feature_atoms

        atom_pos = data.pos.float()
        atom_pos = atom_pos - atom_pos.mean(dim=0)

        data.node_pos = atom_pos

        # Assume data has already been provided with num_nodes, num_bonds, bond_index, bond_type, and diffu_idx initialized.
        # edge_type_mat = torch.zeros([data.num_nodes, data.num_nodes, data.feature_bonds.shape[1]], dtype=torch.long)
        # edge_type_mat[data.bond_index[0], data.bond_index[1]] = data.feature_bonds  # Symmetric filling is needed if undirected
        # print('edge_type_mat previously',edge_type_mat, edge_type_mat.shape)
        # exit()
        edge_type_mat = torch.zeros([data.num_nodes, data.num_nodes, data.feature_bonds.shape[1] + 1], dtype=torch.long)

        # Set the default values for all elements: [0, 0, ..., 0, 1]
        edge_type_mat[..., -1] = 1

        # Update the elements where bonds exist
        bond_features = torch.cat([data.feature_bonds, torch.zeros(data.feature_bonds.shape[0], 1, dtype=torch.long)], dim=1)
        edge_type_mat[data.bond_index[0], data.bond_index[1]] = bond_features
        # print('edge_type_mat latter',edge_type_mat, edge_type_mat.shape)
        # exit()

        # Only extract upper triangle indices (excluding diagonal)
        halfedge_index = torch.triu_indices(data.num_nodes, data.num_nodes, offset=1)
        halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]

        # Create a mask to filter edges either containing diffusion index nodes or having a non-zero type
        is_diffusion_node = torch.zeros(data.num_nodes, dtype=torch.bool)
        try:
            is_diffusion_node[data.diffu_idx] = True
        except:
            print(data)
            traceback.print_exc()
            exit()

        # Check if any of the node pairs contain diffusion index nodes
        mask_diffusion = is_diffusion_node[halfedge_index[0]] | is_diffusion_node[halfedge_index[1]]
        # print(halfedge_type,'halfedge_type')
        # exit()

        # Check for non-zero edge types
        mask_non_zero_type = halfedge_type != 0
        # print(mask_non_zero_type,'mask_non_zero_type before')
        mask_non_zero_type = mask_non_zero_type.any(dim=1)
        # print(mask_non_zero_type,'mask_non_zero_type')
        # exit()

        # Combine masks
        final_mask = mask_diffusion | mask_non_zero_type

        # Apply mask to obtain final indices and types
        non_zero_halfedge_index = halfedge_index[:, final_mask]
        non_zero_halfedge_type = halfedge_type[final_mask]

        # If you need to identify the diffusion-related edges among the filtered results
        mask_diff_bond_type_idx = is_diffusion_node[non_zero_halfedge_index[0]] | is_diffusion_node[non_zero_halfedge_index[1]]

        # The final indices of bonds related to diffusion
        diff_bond_type_idx = torch.nonzero(mask_diff_bond_type_idx).squeeze()
        assert len(non_zero_halfedge_type) == len(non_zero_halfedge_index[0])
        
        data.halfedge_index = non_zero_halfedge_index
        data.halfedge_type = non_zero_halfedge_type
        # print(non_zero_halfedge_index)
        # print(non_zero_halfedge_type)
        # assert False

        bond_mask = torch.ones_like(data.halfedge_type, dtype=torch.bool)
        # bond_mask = initial_bond_mask.expand(-1, self.num_edge_types)
        bond_mask[diff_bond_type_idx] = False
        data.diff_bond_type_idx = bond_mask.clone().detach()

        atom_mask = torch.ones_like(data.node_type, dtype=torch.bool)
        # atom_mask = initial_atom_mask.expand(-1, self.num_node_types)
        atom_mask[data.diffu_idx] = False
        data.diff_idx = atom_mask.clone().detach()

        pos_mask = torch.ones_like(data.node_pos, dtype=torch.bool)
        pos_mask[data.diffu_idx] = False
        data.diff_pos_idx = pos_mask.clone().detach()

        return data
    
    def decode_output(self, pred_node, pred_pos, pred_halfedge, halfedge_index):
        """
        Get the atom and bond information from the prediction (latent space)
        They should be np.array
        pred_node: [n_nodes, n_node_types]
        pred_pos: [n_nodes, 3]
        pred_halfedge: [n_halfedges, n_edge_types]
        """
        # get atom and element
        pred_atom = softmax(pred_node, axis=-1)
        atom_type = np.argmax(pred_atom, axis=-1)
        atom_prob = np.max(pred_atom, axis=-1)
        isnot_masked_atom = (atom_type < self.num_element)
        if not isnot_masked_atom.all():
            edge_index_changer = - np.ones(len(isnot_masked_atom), dtype=np.int64)
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())
        atom_type = atom_type[isnot_masked_atom]
        atom_prob = atom_prob[isnot_masked_atom]
        element = np.array([self.nodetype_to_ele[i] for i in atom_type])
        
        # get pos
        atom_pos = pred_pos[isnot_masked_atom]
        
        # get bond
        if self.num_edge_types == 1:
            return {
                'element': element,
                'atom_pos': atom_pos,
                'atom_prob': atom_prob,
            }
        pred_halfedge = softmax(pred_halfedge, axis=-1)
        edge_type = np.argmax(pred_halfedge, axis=-1)  # omit half for simplicity
        edge_prob = np.max(pred_halfedge, axis=-1)
        
        is_bond = (edge_type > 0) & (edge_type <= self.num_bond_types)  # larger is mask type
        bond_type = edge_type[is_bond]
        bond_prob = edge_prob[is_bond]
        bond_index = halfedge_index[:, is_bond]
        if not isnot_masked_atom.all():
            bond_index = edge_index_changer[bond_index]
            bond_for_masked_atom = (bond_index < 0).any(axis=0)
            bond_index = bond_index[:, ~bond_for_masked_atom]
            bond_type = bond_type[~bond_for_masked_atom]
            bond_prob = bond_prob[~bond_for_masked_atom]

        bond_type = np.concatenate([bond_type, bond_type])
        bond_prob = np.concatenate([bond_prob, bond_prob])
        bond_index = np.concatenate([bond_index, bond_index[::-1]], axis=1)
        
        return {
            'element': element,
            'atom_pos': atom_pos,
            'bond_type': bond_type,
            'bond_index': bond_index,
            
            'atom_prob': atom_prob,
            'bond_prob': bond_prob,
        }
        
    
def make_data_placeholder(n_graphs, device=None, max_size=None):
    # n_nodes_list = np.random.randint(15, 50, n_graphs)
    if max_size is None:  # use statistics from GEOM-Drug dataset
        n_nodes_list = np.random.normal(24.923464980477522, 5.516291901819105, size=n_graphs)
    else:
        n_nodes_list = np.array([max_size] * n_graphs)
    n_nodes_list = n_nodes_list.astype('int64')
    batch_node = np.concatenate([np.full(n_nodes, i) for i, n_nodes in enumerate(n_nodes_list)])
    halfedge_index = []
    batch_halfedge = []
    idx_start = 0
    for i_mol, n_nodes in enumerate(n_nodes_list):
        halfedge_index_this_mol = torch.triu_indices(n_nodes, n_nodes, offset=1)
        halfedge_index.append(halfedge_index_this_mol + idx_start)
        n_edges_this_mol = len(halfedge_index_this_mol[0])
        batch_halfedge.append(np.full(n_edges_this_mol, i_mol))
        idx_start += n_nodes
    
    batch_node = torch.LongTensor(batch_node)
    batch_halfedge = torch.LongTensor(np.concatenate(batch_halfedge))
    halfedge_index = torch.cat(halfedge_index, dim=1)
    
    if device is not None:
        batch_node = batch_node.to(device)
        batch_halfedge = batch_halfedge.to(device)
        halfedge_index = halfedge_index.to(device)
    return {
        # 'n_graphs': n_graphs,
        'batch_node': batch_node,
        'halfedge_index': halfedge_index,
        'batch_halfedge': batch_halfedge,
    }

def make_mydata_placeholder(n_graphs, ref_sdf, device=None, max_size=None):
    # n_nodes_list = np.random.randint(15, 50, n_graphs)
    if max_size is None:  # use statistics from GEOM-Drug dataset
        n_nodes_list = np.random.normal(24.923464980477522, 5.516291901819105, size=n_graphs)
    else:
        n_nodes_list = np.array([max_size] * n_graphs)
    n_nodes_list = n_nodes_list.astype('int64')
    batch_node = np.concatenate([np.full(n_nodes, i) for i, n_nodes in enumerate(n_nodes_list)])
    halfedge_index = []
    batch_halfedge = []
    idx_start = 0
    for i_mol, n_nodes in enumerate(n_nodes_list):
        halfedge_index_this_mol = torch.triu_indices(n_nodes, n_nodes, offset=1)
        halfedge_index.append(halfedge_index_this_mol + idx_start)
        n_edges_this_mol = len(halfedge_index_this_mol[0])
        batch_halfedge.append(np.full(n_edges_this_mol, i_mol))
        idx_start += n_nodes
    
    batch_node = torch.LongTensor(batch_node)
    batch_halfedge = torch.LongTensor(np.concatenate(batch_halfedge))
    halfedge_index = torch.cat(halfedge_index, dim=1)
    
    if device is not None:
        batch_node = batch_node.to(device)
        batch_halfedge = batch_halfedge.to(device)
        halfedge_index = halfedge_index.to(device)
    return {
        # 'n_graphs': n_graphs,
        'batch_node': batch_node,
        'halfedge_index': halfedge_index,
        'batch_halfedge': batch_halfedge,
    }
