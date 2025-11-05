from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import functional as F
from .mytransition import ContigousTransition, GeneralCategoricalTransition
from .mygraph import NodeEdgeNet

from .common import *
from .diffusion import *

def write_xyz_file(pred_pos, element, filename="output_rep_0820_only_lig.xyz"):
    n_atoms = pred_pos.shape[0]  # Number of atoms per molecule
    with open(filename, 'w') as file:
        file.write(f"{n_atoms}\n")
        file.write(f"XYZ file generated from pred_pos and pred_node, index0\n")
        for j in range(n_atoms):
            x, y, z = pred_pos[j]
            file.write(f"{element[j]} {x:.6f} {y:.6f} {z:.6f}\n")


def write_xyz_file_add(pred_pos, element, filename="output_rep_0820_only_lig.xyz"):
    n_atoms = pred_pos.shape[0]  # Number of atoms per molecule
    with open(filename, 'a') as file:
        file.write(f"{n_atoms}\n")
        file.write(f"XYZ file generated from pred_pos and pred_node, index0\n")
        for j in range(n_atoms):
            x, y, z = pred_pos[j]
            file.write(f"{element[j]} {x:.6f} {y:.6f} {z:.6f}\n")

class MolDiff(Module):
    def __init__(self,
        **kwargs
    ):
        super().__init__()
        # print(kwargs)
        self.num_node_types = kwargs['input_node_dim']
        self.num_edge_types = kwargs['input_edge_dim']
        self.bond_len_loss = kwargs['bond_len_loss']

        # # define beta and alpha
        self.define_betas_alphas(kwargs['diff'])

        self.node_embedder = nn.Linear(self.num_node_types, kwargs['node_dim']-kwargs['time_dim']-kwargs['poc_dim'], bias=False)  # element type
        self.edge_embedder = nn.Linear(self.num_edge_types, kwargs['edge_dim']-kwargs['time_dim'], bias=False) # bond type
        self.time_emb = nn.Sequential(
            GaussianSmearing(stop=kwargs['diff']['num_timesteps'], num_gaussians=kwargs['time_dim'], type_='linear'),
        )
        self.poc_emb = nn.Embedding(2, kwargs['poc_dim'])
        
        # # denoiser
        if kwargs['denoiser']['backbone'] == 'NodeEdgeNet':
            self.denoiser = NodeEdgeNet(kwargs['node_dim'], kwargs['edge_dim'], **kwargs['denoiser'])
        else:
            raise NotImplementedError(kwargs['denoiser']['backbone'])

        # # decoder
        self.node_decoder = MLP(kwargs['node_dim'], self.num_node_types, kwargs['node_dim'])
        self.edge_decoder = MLP(kwargs['edge_dim'], self.num_edge_types, kwargs['edge_dim'])


    def define_betas_alphas(self, config):
        self.num_timesteps = config['num_timesteps']
        

        self.scaling = getattr(config, 'scaling', [1., 1., 1.])
        # # diffusion for pos
        pos_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config['diff_pos']
        )
        assert self.scaling[0] == 1, 'scaling for pos should be 1'
        self.pos_transition = ContigousTransition(pos_betas)

        # # diffusion for node type
        node_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config['diff_atom']
        )

        scaling_node = self.scaling[1]
        # print('scaling_node', scaling_node)
        self.node_transition = ContigousTransition(node_betas, self.num_node_types, scaling_node)

        # # diffusion for edge type
        edge_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config['diff_bond']
        )

        scaling_edge = self.scaling[2]
        self.edge_transition = ContigousTransition(edge_betas, self.num_edge_types, scaling_edge)

    def sample_time(self, num_graphs, device, **kwargs):
        # sample time
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt

    def add_noise(self, node_type, node_pos, batch_node,
                    halfedge_type, halfedge_index, batch_halfedge,
                    num_mol, t, bond_predictor=None, **kwargs):
            num_graphs = num_mol
            device = node_pos.device

            time_step = t * torch.ones(num_graphs, device=device).long()

            # 2.1 perturb pos, node, edge
            pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
            node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
            halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
            h_node_pert, h_node_0 = node_pert
            h_halfedge_pert, h_halfedge_0 = halfedge_pert
            return [h_node_pert, pos_pert, h_halfedge_pert]

    def get_loss(self, node_type, node_pos, batch_node,
                halfedge_type, halfedge_index, batch_halfedge,
                num_mol, diff_idx, diff_pos_idx, diff_bond_type_idx, poc_or_not, **kwargs
    ):
        num_graphs = num_mol
        device = node_pos.device

        # 1. sample noise levels
        time_step, _ = self.sample_time(num_graphs, device)

        # 2.1 perturb pos, node, edge
        # only when the node is at diffu_idx, the pos of node is at diffu_idx and the edge's two node are all at diffu_idx, the pertubation works
        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node, diff_pos_idx)
        # print(node_type,'node_type in model',node_type.shape)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node, diff_idx)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge, diff_bond_type_idx)
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

        h_node_pert, h_node_0 = node_pert
        h_halfedge_pert, h_halfedge_0 = halfedge_pert
        
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)

        # 3. forward to denoise
        preds = self(
            h_node_pert, pos_pert, batch_node,
            h_edge_pert, edge_index, batch_edge, poc_or_not, 
            time_step, 
        )
        pred_node = preds['pred_node']
        pred_pos = preds['pred_pos']
        pred_halfedge = preds['pred_halfedge']

        mask_node = ~diff_idx.all(dim=1)
        mask_pos = ~diff_pos_idx.all(dim=1)
        mask_half_edge = ~diff_bond_type_idx.all(dim=1)

        # node_for_loss = h_node_0[mask_node]
        # pred_node_for_loss = pred_node[mask_node]
        # pos_for_loss = node_pos[mask_pos]
        # pred_pos_for_loss = pred_pos[mask_pos]
        # halfedge_for_loss = h_halfedge_0[mask_half_edge]
        # pred_halfedge_for_loss = pred_halfedge[mask_half_edge]


        node_for_loss = h_node_0
        pred_node_for_loss = pred_node
        pos_for_loss = node_pos
        pred_pos_for_loss = pred_pos
        halfedge_for_loss = h_halfedge_0
        pred_halfedge_for_loss = pred_halfedge

        return node_for_loss, pred_node_for_loss, pos_for_loss, pred_pos_for_loss, halfedge_for_loss, pred_halfedge_for_loss



    def forward(self, 
                h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, poc_or_not, 
                t):
        """
        Predict Mol at step `0` given perturbed Mol at step `t` with hidden dims and time step
        """
        # 1 node and edge embedding + time embedding
        time_embed_node = self.time_emb(t.index_select(0, batch_node))
        poc_embed_node = self.poc_emb(poc_or_not)
        dtype = self.node_embedder.weight.dtype
        device = self.node_embedder.weight.device
        # print(h_node_pert.shape, h_node_pert, poc_or_not)
        mask = poc_or_not == 1
        selected_h_node_pert = h_node_pert[mask]
        h_node_pert = h_node_pert.to(device=device, dtype=dtype)
        time_embed_node = time_embed_node.to(device=device, dtype=dtype)
        poc_embed_node = poc_embed_node.to(device=device, dtype=dtype)
        h_node_pert = torch.cat([self.node_embedder(h_node_pert), time_embed_node, poc_embed_node], dim=-1)
        time_embed_edge = self.time_emb(t.index_select(0, batch_edge))
        h_edge_pert = torch.cat([self.edge_embedder(h_edge_pert), time_embed_edge], dim=-1)

        # 2 diffuse to get the updated node embedding and bond embedding
        h_node, pos_node, h_edge = self.denoiser(
            h_node=h_node_pert,
            pos_node=pos_pert, 
            h_edge=h_edge_pert, 
            edge_index=edge_index,
            node_time=t.index_select(0, batch_node).unsqueeze(-1) / self.num_timesteps,
            edge_time=t.index_select(0, batch_edge).unsqueeze(-1) / self.num_timesteps,
        )
        
        n_halfedges = h_edge.shape[0] // 2
        pred_node = self.node_decoder(h_node)
        pred_halfedge = self.edge_decoder(h_edge[:n_halfedges]+h_edge[n_halfedges:])
        pred_pos = pos_node
        
        return {
            'pred_node': pred_node,
            'pred_pos': pred_pos,
            'pred_halfedge': pred_halfedge,
        }  # at step 0

    @torch.no_grad()
    def sample(self, n_graphs, batch_node, halfedge_index, 
               batch_halfedge, ref_data, traj_fn,
                bond_predictor=None, guidance=None):
        device = batch_node.device
        # # 1. get the init values (position, node types)
        # n_graphs = len(n_nodes_list)
        n_nodes_all = len(batch_node)
        n_halfedges_all = len(batch_halfedge)
        # print('n_nodes_all', n_nodes_all, 'n_halfedges_all', n_halfedges_all)
        node_init1 = self.node_transition.sample_init(n_nodes_all)
        pos_init1 = self.pos_transition.sample_init([n_nodes_all, 3])
        halfedge_init1 = self.edge_transition.sample_init(n_halfedges_all)

        ref_data = ref_data.to(node_init1.device)
        node_init = torch.where(ref_data.atom_mask, ref_data.node_padding, node_init1)
        # print(node_init, 'node_init')


        pos_init = torch.where(ref_data.pos_mask, ref_data.pos_padding, pos_init1)

        halfedge_init = torch.where(ref_data.bond_mask, ref_data.half_edge_padding, halfedge_init1)

        h_node_init = node_init
        h_halfedge_init = halfedge_init
            

        # # 1.5 log init
        node_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, h_node_init.shape[-1]],
                                dtype=h_node_init.dtype).to(device)
        pos_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, 3], dtype=pos_init.dtype).to(device)
        halfedge_traj = torch.zeros([self.num_timesteps+1, n_halfedges_all, h_halfedge_init.shape[-1]],
                                    dtype=h_halfedge_init.dtype).to(device)
        node_traj[0] = h_node_init
        pos_traj[0] = pos_init
        halfedge_traj[0] = h_halfedge_init

        # # 2. sample loop
        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
            
            # # 1 inference
            preds = self(
                h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, ref_data.pocket_or_not, 
                time_step, 
            )
            pred_node = preds['pred_node']  # (N, num_node_types)
            pred_pos = preds['pred_pos']  # (N, 3)
            pred_halfedge = preds['pred_halfedge']  # (E//2, num_bond_types)
            # pred_node_for_xyz = pred_node.cpu().detach().numpy()
            # atom_type_pred = np.argmax(pred_node_for_xyz[:, :8], axis=-1)
            # atom_list = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
            # element_pred = np.array([atom_list[i] for i in atom_type_pred])
            # write_xyz_file_add(pred_pos, element_pred, f'{traj_fn}')

            # make only the diffu node, pos, edge update, but not all the nodes, pos, edges
            pred_node = torch.where(ref_data.atom_mask, ref_data.node_padding, pred_node)
            pred_pos = torch.where(ref_data.pos_mask, ref_data.pos_padding, pred_pos)
            # print(pred_pos,' pred_pos')
            pred_halfedge = torch.where(ref_data.bond_mask, ref_data.half_edge_padding, pred_halfedge)

            # # 2 get the t - 1 state
            # pos 
            pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, batch=batch_node)
            # print(pos_prev, 'pos_prev')
            pos_prev = torch.where(ref_data.pos_mask, ref_data.pos_padding, pos_prev)

            h_node_prev = self.node_transition.get_prev_from_recon(
                x_t=h_node_pert, x_recon=pred_node, t=time_step, batch=batch_node)
            h_node_prev = torch.where(ref_data.atom_mask, ref_data.node_padding, h_node_prev)
            h_halfedge_prev = self.edge_transition.get_prev_from_recon(
                x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, batch=batch_halfedge)
            h_halfedge_prev = torch.where(ref_data.bond_mask, ref_data.half_edge_padding, h_halfedge_prev)


            # log update
            node_traj[i+1] = h_node_prev
            pos_traj[i+1] = pos_prev
            halfedge_traj[i+1] = h_halfedge_prev

            # # 3 update t-1
            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev

        # # 3. get the final positions

        # only return the pred node, where ref_data_mask == False
        # pred_node = torch.where(ref_data.atom_mask, node_init, node_traj[-1])
        # pred_pos = torch.where(ref_data.pos_mask, pos_init, pos_traj[-1])
        # pred_halfedge = torch.where(ref_data.bond_mask, halfedge_init, halfedge_traj[-1])
        # exit()
        # print(node_traj[-1], ref_data.atom_mask, node_traj[-1].shape,  ref_data.atom_mask.shape)
        pred_node = node_traj[-1][~ref_data.atom_mask_recon].reshape(-1, ref_data.atom_mask_recon.shape[-1])
        pred_pos = pos_traj[-1][~ref_data.pos_mask_recon].reshape(-1, ref_data.pos_mask_recon.shape[-1])
        pred_halfedge = torch.where(ref_data.bond_mask_recon, ref_data.half_edge_padding, halfedge_traj[-1])
        origin_node = node_traj[-1][ref_data.atom_mask_recon].reshape(-1, ref_data.atom_mask_recon.shape[-1])
        origin_pos = pos_traj[-1][ref_data.pos_mask_recon].reshape(-1, ref_data.pos_mask_recon.shape[-1])
        # pred_halfedge = halfedge_traj[-1][~ref_data.bond_mask].reshape(-1, ref_data.bond_mask.shape[-1])
        # print(pred_node.shape, pred_pos.shape, pred_halfedge.shape, ref_data.bond_mask.shape, 'pred_node, pred_pos, pred_halfedge,')

        return {
            'pred': [pred_node, pred_pos, pred_halfedge],
            'traj': [node_traj, pos_traj, halfedge_traj],
            'origin': [origin_node, origin_pos, pred_halfedge],
        }


