import torch
import torch.nn as nn
import numpy as np
import pickle
import traceback
from rdkit import Chem
from rdkit import Geometry
from copy import deepcopy
import itertools
import re
# from .layers.embed import NonLinear
from .representation import MolComp
from easydict import EasyDict



class MolWrapper(nn.Module):
    def __init__(
        self,
        **layer_configs
    ):
        super().__init__()
        print(layer_configs)
        self.model = MolComp(**layer_configs)
        self.pos_noise_std = layer_configs['pos_noise_std']
        self.num_mols = layer_configs['sample']['num_mols']
        self.guidance = layer_configs['sample']['guidance']

    def forward(self, batch):
        # print(batch)
        pos_noise = torch.randn_like(batch.node_pos) * self.pos_noise_std
        loss= self.model.get_loss(
                # compose
                node_type = batch.node_type,
                node_pos = batch.node_pos + pos_noise,
                batch_node = batch.node_type_batch,
                halfedge_type = batch.halfedge_type,
                halfedge_index = batch.halfedge_index,
                batch_halfedge = batch.halfedge_type_batch,
                num_mol = batch.num_graphs,
                # diffu
                diff_idx = batch.diff_idx,
                diff_pos_idx = batch.diff_pos_idx,
                diff_bond_type_idx = batch.diff_bond_type_idx,
                poc_or_not = batch.pocket_or_not
            )
        return loss

    def sample(self, batch):
        pool = EasyDict({
            'failed': [],
            'finished': [],
        })
        batch_size = len(batch.name)
        n_graphs = min(batch_size, (self.num_mols - len(pool.finished))*2)
        batch_holder = self.make_mydata_placeholder(n_graphs=n_graphs, ref_data = batch, device=batch.pos.device, max_size=None)
        bond_predictor = None
        while len(pool.finished) < self.num_mols:
            batch_node, halfedge_index, batch_halfedge, ref_data = batch_holder['batch_node'], batch_holder['halfedge_index'], batch_holder['batch_halfedge'], batch_holder['ref_data']
            outputs = self.model.sample(
                n_graphs=n_graphs,
                batch_node=batch_node,
                halfedge_index=halfedge_index,
                batch_halfedge=batch_halfedge,
                ref_data=ref_data,
                bond_predictor=bond_predictor,
                guidance=self.guidance,
            )
            outputs = {key:[v.cpu().numpy() for v in value] for key, value in outputs.items()}
            batch_node, halfedge_index, batch_halfedge = batch_node.cpu().numpy(), halfedge_index.cpu().numpy(), batch_halfedge.cpu().numpy()
            try:
                output_list = self.seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
            except:
                continue
            gen_list = []
            for i_mol, output_mol in enumerate(output_list):
                mol_info = self.decode_output(
                    pred_node=output_mol['pred'][0],
                    pred_pos=output_mol['pred'][1],
                    pred_halfedge=output_mol['pred'][2],
                    halfedge_index=output_mol['halfedge_index'],
                )  # note: traj is not used
                try:
                    rdmol = self.reconstruct_from_generated_with_edges(mol_info)
                except:
                    traceback.print_exc()
                    pool.failed.append(mol_info)
                    print('Reconstruction error encountered.')
                    continue
                mol_info['rdmol'] = rdmol
                smiles = Chem.MolToSmiles(rdmol)
                mol_info['smiles'] = smiles
                if '.' in smiles:
                    print('Incomplete molecule: %s' % smiles)
                    pool.failed.append(mol_info)
                else:   # Pass checks!
                    print('Success: %s' % smiles)
                    p_save_traj = np.random.rand()  # save traj
                    if p_save_traj <  0.02:
                        traj_info = [self.decode_output(
                            pred_node=output_mol['traj'][0][t],
                            pred_pos=output_mol['traj'][1][t],
                            pred_halfedge=output_mol['traj'][2][t],
                            halfedge_index=output_mol['halfedge_index'],
                        ) for t in range(len(output_mol['traj'][0]))]
                        mol_traj = []
                        for t in range(len(traj_info)):
                            try:
                                mol_traj.append(self.reconstruct_from_generated_with_edges(traj_info[t], False, add_edge=add_edge))
                            except:
                                mol_traj.append(Chem.MolFromSmiles('O'))
                        mol_info['traj'] = mol_traj
                    gen_list.append(mol_info)

    def make_mydata_placeholder(self, n_graphs, ref_data, device=None, max_size=None):
        if max_size is None:  # use statistics from GEOM-Drug dataset
            n_nodes_list = np.random.normal(24.923464980477522, 5.516291901819105, size=n_graphs)
            n_nodes_list = np.random.normal(9, 0, size=n_graphs)
        else:
            n_nodes_list = np.array([max_size] * n_graphs)
        n_nodes_list = n_nodes_list.astype('int64')
        batch_node = np.concatenate([np.full(n_nodes + ref_data.num_nodes, i) for i, n_nodes in enumerate(n_nodes_list)])
        halfedge_index = []
        batch_halfedge = []
        idx_start = 0
        diffu_idx = []
        node_padding = []
        pos_padding = []
        half_edge_padding = []



        for i_mol, n_nodes in enumerate(n_nodes_list):
            halfedge_index_this_mol = torch.triu_indices(n_nodes + ref_data.num_nodes, 
                                                        n_nodes + ref_data.num_nodes, offset=1)
            # print(halfedge_index_this_mol, 'halfedge_index_this_mol', idx_start, halfedge_index_this_mol + idx_start)
            halfedge_index.append(halfedge_index_this_mol + idx_start)
            # print(halfedge_index,'halfedge_index')
            n_edges_this_mol = len(halfedge_index_this_mol[0])
            batch_halfedge.append(np.full(n_edges_this_mol, i_mol))
            for i in range(n_nodes):
                diffu_idx.append(i + idx_start)
            idx_start += (n_nodes + ref_data.num_nodes)
            # print(diffu_idx, 'diffu_idx')
            # ref data to the device of zero_padding

            node_padding_init = torch.randn(n_nodes,29)
            ref_data = ref_data.to(node_padding_init.device)
            type_padded = torch.cat([node_padding_init, ref_data.node_type])
            node_padding.append(type_padded)

            zero_pos_padding = torch.zeros(n_nodes, 3)
            pos_padded = torch.cat([zero_pos_padding, ref_data.node_pos])
            pos_padding.append(pos_padded)


            this_mol_halfedge_index = halfedge_index_this_mol + idx_start
            this_mol_halfedge_index_T = this_mol_halfedge_index.T
            ref_mol_halfedge_index_T = ref_data.halfedge_index.T + idx_start
            ref_index_to_type = {tuple(value.tolist()): i for i, value in enumerate(ref_mol_halfedge_index_T)}

            long_edge_types = []

            for ix, edge in enumerate(this_mol_halfedge_index_T):
                edge_tuple = tuple(edge.tolist())
                
                if edge_tuple in ref_index_to_type:
                    type_index = ref_index_to_type[edge_tuple]
                    # print(ref_data.halfedge_type[type_index])
                    long_edge_types.append(ref_data.halfedge_type[type_index])
                else:
                    long_edge_types.append(torch.tensor([0,0,0,0,0,0,0]))

            long_edge_types = torch.stack(long_edge_types)

            half_edge_padding.append(long_edge_types)

        batch_node = torch.LongTensor(batch_node)
        batch_halfedge = torch.LongTensor(np.concatenate(batch_halfedge))
        halfedge_index = torch.cat(halfedge_index, dim=1)



        diffu_idx_tensor = torch.tensor(diffu_idx)

        # Transpose halfedge_index for easier comparison
        halfedge_index_T = halfedge_index.T

        # Check if any element in halfedge_index_T is in diffu_idx_tensor
        mask_0 = torch.isin(halfedge_index_T[:, 0], diffu_idx_tensor)
        mask_1 = torch.isin(halfedge_index_T[:, 1], diffu_idx_tensor)

        # Combine the masks using logical OR
        mask = mask_0 | mask_1

        # Get the indices where the condition is True
        diff_bond_type_idx = torch.nonzero(mask).squeeze()

        # Convert to a list if needed
        diff_bond_type_idx = diff_bond_type_idx.tolist()
    

        initial_bond_mask = torch.ones_like(batch_halfedge, dtype=torch.bool)
        initial_bond_mask = initial_bond_mask.unsqueeze(1)
        bond_mask = initial_bond_mask.expand(-1, ref_data.halfedge_type.shape[1])
        # print(bond_mask, 'bond_mask')
        bond_mask = bond_mask.clone()  # Clone the tensor before modification
        bond_mask[diff_bond_type_idx] = False
        # print(bond_mask, 'bond_mask')
        ref_data.bond_mask = bond_mask.clone().detach()

        initial_atom_mask = torch.ones_like(batch_node, dtype=torch.bool)
        initial_atom_mask = initial_atom_mask.unsqueeze(1)
        atom_mask = initial_atom_mask.expand(-1, ref_data.node_type.shape[1])
        atom_mask = atom_mask.clone()
        atom_mask[diffu_idx] = False
        ref_data.atom_mask = atom_mask.clone().detach()

        init_pos_mask = torch.ones_like(batch_node, dtype=torch.bool)
        init_pos_mask = init_pos_mask.unsqueeze(1)
        pos_mask = init_pos_mask.expand(-1, 3)
        pos_mask = pos_mask.clone()
        pos_mask[diffu_idx] = False
        ref_data.pos_mask = pos_mask.clone().detach()


        ref_data.node_padding = torch.cat(node_padding, dim=0)
        ref_data.pos_padding = torch.cat(pos_padding, dim=0)
        ref_data.half_edge_padding = torch.cat(half_edge_padding, dim=0)
        ref_data.pocket_or_not = torch.cat([torch.zeros(n_nodes), ref_data.pocket_or_not], dim=0)
        ref_data.pocket_or_not = ref_data.pocket_or_not.long()

        if device is not None:
            batch_node = batch_node.to(device)
            batch_halfedge = batch_halfedge.to(device)
            halfedge_index = halfedge_index.to(device)
        return {
            # 'n_graphs': n_graphs,
            'batch_node': batch_node,
            'halfedge_index': halfedge_index,
            'batch_halfedge': batch_halfedge,
            'ref_data': ref_data,

        }

    def seperate_outputs(self, outputs, n_graphs, batch_node, halfedge_index, batch_halfedge):
        outputs_pred = outputs['pred']
        outputs_traj = outputs['traj']

        new_outputs = []
        for i_mol in range(n_graphs):
            ind_node = (batch_node == i_mol)
            ind_halfedge = (batch_halfedge == i_mol)
            assert ind_node.sum() * (ind_node.sum()-1) == ind_halfedge.sum() * 2
            new_pred_this = [outputs_pred[0][ind_node],  # node type
                            outputs_pred[1][ind_node],  # node pos
                            outputs_pred[2][ind_halfedge]]  # halfedge type
                            
            new_traj_this = [outputs_traj[0][:, ind_node],  # node type. The first dim is time
                            outputs_traj[1][:, ind_node],  # node pos
                            outputs_traj[2][:, ind_halfedge]]  # halfedge type
            
            halfedge_index_this = halfedge_index[:, ind_halfedge]
            assert ind_node.nonzero()[0].min() == halfedge_index_this.min()
            halfedge_index_this = halfedge_index_this - ind_node.nonzero()[0].min()

            new_outputs.append({
                'pred': new_pred_this,
                'traj': new_traj_this,
                'halfedge_index': halfedge_index_this,
            })
        return new_outputs


    def reconstruct_from_generated_with_edges(self, mol_info, check_validity=True, add_edge=None):
        xyz = mol_info['atom_pos'].tolist()
        atomic_nums = mol_info['element'].tolist()
        bond_index = mol_info['bond_index'].tolist()
        bond_type = mol_info['bond_type'].tolist()
        n_atoms = len(atomic_nums)

        rd_mol = Chem.RWMol()
        rd_conf = Chem.Conformer(n_atoms)
        
        # add atoms and coordinates
        for i, atom in enumerate(atomic_nums):
            rd_atom = Chem.Atom(atom)
            rd_mol.AddAtom(rd_atom)
            rd_coords = Geometry.Point3D(*xyz[i])
            rd_conf.SetAtomPosition(i, rd_coords)
        rd_mol.AddConformer(rd_conf)
        
        # add bonds
        for i, type_this in enumerate(bond_type):
            node_i, node_j = bond_index[0][i], bond_index[1][i]
            if node_i < node_j:
                if type_this == 0:
                    rd_mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
                elif type_this == 1:
                    rd_mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
                elif type_this == 2:
                    rd_mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
                elif type_this == 3:
                    rd_mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
                elif type_this == 4:
                    # non colvent bond
                    pass
                else:
                    raise Exception('unknown bond order {}'.format(type_this))
        
        
        mol = rd_mol.GetMol()
        if check_validity:
            try:
                Chem.SanitizeMol(mol)
                fixed = True
            except Exception as e:
                fixed = False
            
            if not fixed:
                try:
                    Chem.Kekulize(deepcopy(mol))
                except Chem.rdchem.KekulizeException as e:
                    err = e
                    if 'Unkekulized' in err.args[0]:
                        mol, fixed = self.fix_aromatic(mol)

            # valence error for N 
            if not fixed:
                mol, fixed = self.fix_valence(mol)
                
            if not fixed:
                mol, fixed = self.fix_aromatic(mol, True)
                
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                raise ValueError('Cannot SanitizeMol')
                # return None
        return mol

    def decode_output(self, pred_node, pred_pos, pred_halfedge, halfedge_index):
        """
        Get the atom and bond information from the prediction (latent space)
        They should be np.array
        pred_node: [n_nodes, n_node_types]
        pred_pos: [n_nodes, 3]
        pred_halfedge: [n_halfedges, n_edge_types]
        """
        # get atom and element

        atom_type = np.argmax(pred_node[: , :8], axis=-1)
        atom_prob = np.max(pred_node[: , :8], axis=-1)
        atom_list = ['C','N','O','F','P','S','Cl','Br']
        element = np.array([atom_list[i] for i in atom_type])
        
        # get pos
        atom_pos = pred_pos
        
        edge_type = np.argmax(pred_halfedge, axis=-1)  # omit half for simplicity
        edge_prob = np.max(pred_halfedge, axis=-1)
        sum_prob = np.sum(pred_halfedge, axis=-1) > 0.5
        
        is_bond = (edge_type < 3) & sum_prob
        bond_type = edge_type[is_bond]
        bond_prob = edge_prob[is_bond]
        bond_index = halfedge_index[:, is_bond]

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

    def fix_valence(self, mol):
        mol = deepcopy(mol)
        fixed = False
        cnt_loop = 0
        while True:
            try:
                Chem.SanitizeMol(mol)
                fixed = True
                break
            except Chem.rdchem.AtomValenceException as e:
                err = e
            except Exception as e:
                return mol, False # from HERE: rerun sample
            cnt_loop += 1
            if cnt_loop > 100:
                break
            N4_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
            index = N4_valence.findall(err.args[0])
            if len(index) > 0:
                mol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
        return mol, fixed

    def fix_aromatic(self, mol, strict=False):
        mol_orig = mol
        atomatic_list = [a.GetIdx() for a in mol.GetAromaticAtoms()]
        N_ring_list = []
        S_ring_list = []
        for ring_sys in self.get_ring_sys(mol):
            if set(ring_sys).intersection(set(atomatic_list)):
                idx_N = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'N']
                if len(idx_N) > 0:
                    idx_N.append(-1) # -1 for not add to this loop
                    N_ring_list.append(idx_N)
                idx_S = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'S']
                if len(idx_S) > 0:
                    idx_S.append(-1) # -1 for not add to this loop
                    S_ring_list.append(idx_S)
        # enumerate S
        fixed = False
        if strict:
            S_ring_list = [s for ring in S_ring_list for s in ring if s != -1]
            permutation = self.get_all_subsets(S_ring_list)
        else:
            permutation = list(itertools.product(*S_ring_list))
        for perm in permutation:
            mol = deepcopy(mol_orig)
            perm = [x for x in perm if x != -1]
            for idx in perm:
                mol.GetAtomWithIdx(idx).SetFormalCharge(1)
            try:
                if strict:
                    mol, fixed = self.fix_valence(mol)
                Chem.SanitizeMol(mol)
                fixed = True
                break
            except:
                continue
        # enumerate N
        if not fixed:
            if strict:
                N_ring_list = [s for ring in N_ring_list for s in ring if s != -1]
                permutation = self.get_all_subsets(N_ring_list)
            else:
                permutation = list(itertools.product(*N_ring_list))
            for perm in permutation:  # each ring select one atom
                perm = [x for x in perm if x != -1]
                # print(perm)
                actions = itertools.product([0, 1], repeat=len(perm))
                for action in actions: # add H or charge
                    mol = deepcopy(mol_orig)
                    for idx, act_atom in zip(perm, action):
                        if act_atom == 0:
                            mol.GetAtomWithIdx(idx).SetNumExplicitHs(1)
                        else:
                            mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    try:
                        if strict:
                            mol, fixed = self.fix_valence(mol)
                        Chem.SanitizeMol(mol)
                        fixed = True
                        break
                    except:
                        continue
                if fixed:
                    break
        return mol, fixed

    def get_ring_sys(self, mol):
        all_rings = Chem.GetSymmSSSR(mol)
        if len(all_rings) == 0:
            ring_sys_list = []
        else:
            ring_sys_list = [all_rings[0]]
            for ring in all_rings[1:]:
                form_prev = False
                for prev_ring in ring_sys_list:
                    if set(ring).intersection(set(prev_ring)):
                        prev_ring.extend(ring)
                        form_prev = True
                        break
                if not form_prev:
                    ring_sys_list.append(ring)
        ring_sys_list = [list(set(x)) for x in ring_sys_list]
        return ring_sys_list

    def get_all_subsets(self, ring_list):
        all_sub_list = []
        for n_sub in range(len(ring_list)+1):
            all_sub_list.extend(itertools.combinations(ring_list, n_sub))
        return all_sub_list