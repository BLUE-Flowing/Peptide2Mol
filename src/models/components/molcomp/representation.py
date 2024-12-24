from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import functional as F
from .mytransition import ContigousTransition, GeneralCategoricalTransition
from .mygraph import NodeEdgeNet
import torch_scatter
from .common import *
from .diffusion import *


class MolComp(Module):
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
                num_mol, diff_idx, diff_pos_idx, diff_bond_type_idx, poc_or_not, **kwargs):
        num_graphs = num_mol
        device = node_pos.device

        # 1. Sample noise levels
        time_step, _ = self.sample_time(num_graphs, device)

        # 2. Generate Positive and Negative Masks using Subgraph Sampling
        pos_mask, neg_mask = self.get_pos_neg_mask(batch_node)
        # Apply subgraph sampling to create positive and negative subgraphs
        pos_node_type, pos_node_pos, pos_batch_node, pos_halfedge_type, pos_halfedge_index, pos_batch_halfedge, pos_diff_idx, pos_diff_pos_idx, pos_diff_bond_type_idx = \
            self.subgraph_sample(node_type, node_pos, batch_node, halfedge_type, halfedge_index, batch_halfedge, pos_mask, diff_idx, diff_pos_idx, diff_bond_type_idx)
        neg_node_type, neg_node_pos, neg_batch_node, neg_halfedge_type, neg_halfedge_index, neg_batch_halfedge, neg_diff_idx, neg_diff_pos_idx, neg_diff_bond_type_idx = \
            self.subgraph_sample(node_type, node_pos, batch_node, halfedge_type, halfedge_index, batch_halfedge, neg_mask, diff_idx, diff_pos_idx, diff_bond_type_idx)
        
        pos_poc_or_not = poc_or_not[pos_mask]
        
        pos1_mask, anc_mask = self.get_pos_anc_mask(pos_batch_node)
        pos1_node_type, pos1_node_pos, pos1_batch_node, pos1_halfedge_type, pos1_halfedge_index, pos1_batch_halfedge, pos1_diff_idx, pos1_diff_pos_idx, pos1_diff_bond_type_idx = \
            self.subgraph_sample(pos_node_type, pos_node_pos, pos_batch_node, pos_halfedge_type, pos_halfedge_index, pos_batch_halfedge, pos1_mask, pos_diff_idx, pos_diff_pos_idx, pos_diff_bond_type_idx)
        anc_node_type, anc_node_pos, anc_batch_node, anc_halfedge_type, anc_halfedge_index, anc_batch_halfedge, anc_diff_idx, anc_diff_pos_idx, anc_diff_bond_type_idx = \
            self.subgraph_sample(pos_node_type, pos_node_pos, pos_batch_node, pos_halfedge_type, pos_halfedge_index, pos_batch_halfedge, anc_mask, pos_diff_idx, pos_diff_pos_idx, pos_diff_bond_type_idx)

        # 3. Encode Positive Samples
        pos_embeddings = self.encode(
            pos1_node_type, pos1_node_pos, pos1_batch_node,
            pos1_halfedge_type, pos1_halfedge_index, pos1_batch_halfedge,
            time_step, pos_poc_or_not[pos1_mask], pos1_diff_idx, pos1_diff_pos_idx, pos1_diff_bond_type_idx
        )

        # 4. Encode Negative Samples
        neg_embeddings = self.encode(
            neg_node_type, neg_node_pos, neg_batch_node,
            neg_halfedge_type, neg_halfedge_index, neg_batch_halfedge,
            time_step, poc_or_not[neg_mask], neg_diff_idx, neg_diff_pos_idx, neg_diff_bond_type_idx
        )

        # 5. Encode Original Samples (Anchor)
        anchor_embeddings = self.encode(
            anc_node_type, anc_node_pos, anc_batch_node,
            anc_halfedge_type, anc_halfedge_index, anc_batch_halfedge,
            time_step, pos_poc_or_not[anc_mask], anc_diff_idx, anc_diff_pos_idx, anc_diff_bond_type_idx
        )

        # 6. Compute Contrastive Loss
        loss = self.compute_contrastive_loss(anchor_embeddings, pos_embeddings, neg_embeddings)

        return loss

    def get_pos_neg_mask(self, batch_node):
        # Get the number of nodes
        num_nodes = batch_node.size(0)
        # Initialize masks
        pos_mask = torch.zeros(num_nodes, dtype=torch.bool, device=batch_node.device)
        neg_mask = torch.zeros(num_nodes, dtype=torch.bool, device=batch_node.device)
        while True:
            rand_number = torch.rand(num_nodes, device=batch_node.device)
            pos_mask = rand_number < 0.6666
            neg_mask = ~pos_mask
            # Ensure each graph has at least one node in pos and neg masks
            pos_counts = torch_scatter.scatter_sum(pos_mask.long(), batch_node, dim=0)
            neg_counts = torch_scatter.scatter_sum(neg_mask.long(), batch_node, dim=0)
            if torch.all(pos_counts > 0) and torch.all(neg_counts > 0):
                break
        return pos_mask, neg_mask


    def get_pos_anc_mask(self, batch_node):
        # Get the number of nodes
        num_nodes = batch_node.size(0)
        # Initialize masks
        pos_mask = torch.zeros(num_nodes, dtype=torch.bool, device=batch_node.device)
        neg_mask = torch.zeros(num_nodes, dtype=torch.bool, device=batch_node.device)
        while True:
            rand_number = torch.rand(num_nodes, device=batch_node.device)
            pos_mask = rand_number < 0.5
            rand_number = torch.rand(num_nodes, device=batch_node.device)
            neg_mask = rand_number < 0.5
            # Ensure each graph has at least one node in pos and neg masks
            pos_counts = torch_scatter.scatter_sum(pos_mask.long(), batch_node, dim=0)
            neg_counts = torch_scatter.scatter_sum(neg_mask.long(), batch_node, dim=0)
            if torch.all(pos_counts > 0) and torch.all(neg_counts > 0):
                break
        return pos_mask, neg_mask


    def subgraph_sample(self, node_type, node_pos, batch_node, halfedge_type, halfedge_index, batch_halfedge, mask, diff_idx, diff_pos_idx, diff_bond_type_idx):
        # Filter nodes
        sub_node_type = node_type[mask]
        sub_node_pos = node_pos[mask]
        sub_batch_node = batch_node[mask]
        sub_diff_idx = diff_idx[mask]
        sub_diff_pos_idx = diff_pos_idx[mask]
        # Map old node indices to new ones
        idx_mapping = torch.zeros(node_type.size(0), dtype=torch.long, device=node_type.device) - 1
        idx_mapping[mask] = torch.arange(mask.sum(), device=node_type.device)

        # Filter edges
        edge_mask = mask[halfedge_index[0]] & mask[halfedge_index[1]]
        sub_halfedge_type = halfedge_type[edge_mask]
        sub_halfedge_index = halfedge_index[:, edge_mask]
        sub_halfedge_index = idx_mapping[sub_halfedge_index]
        sub_batch_halfedge = batch_halfedge[edge_mask]

        sub_diff_bond_type_idx = diff_bond_type_idx[edge_mask]


        return sub_node_type, sub_node_pos, sub_batch_node, sub_halfedge_type, sub_halfedge_index, sub_batch_halfedge, sub_diff_idx, sub_diff_pos_idx, sub_diff_bond_type_idx


    def encode(self, node_type, node_pos, batch_node,
            halfedge_type, halfedge_index, batch_halfedge,
            time_step, poc_or_not, diff_idx, diff_pos_idx, diff_bond_type_idx):
        # Perturb the inputs
        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node, diff_pos_idx)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node, diff_idx)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge, diff_bond_type_idx)
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

        h_node_pert, _ = node_pert
        h_halfedge_pert, _ = halfedge_pert
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)

        # Forward pass through the model
        preds = self(
            h_node_pert, pos_pert, batch_node,
            h_edge_pert, edge_index, batch_edge, poc_or_not,
            time_step,
        )
        # Obtain embeddings (e.g., from the last layer or a pooling layer)
        embeddings = self.get_graph_embeddings(preds, batch_node)
        return embeddings

    def get_graph_embeddings(self, preds, batch_node):
        # Assuming preds['pred_node'] contains node features
        node_embeddings = preds['pred_node']
        # Aggregate node embeddings to graph embeddings (e.g., mean pooling)
        graph_embeddings = torch_scatter.scatter_mean(node_embeddings, batch_node, dim=0)
        return graph_embeddings


    def compute_contrastive_loss(self, anchor_embeddings, pos_embeddings, neg_embeddings):


        pos_dist, neg_dist = torch.norm(anchor_embeddings - pos_embeddings, dim=-1), torch.norm(anchor_embeddings - neg_embeddings, dim=-1)
        # # d-prime loss  neg_thresh = 0-10
        neg_thresh = 10
        loss = torch.mean(pos_dist, dim=-1) + torch.mean(F.relu(neg_thresh - neg_dist), dim=-1)
        return loss

        # d-prime loss
        # loss = torch.std(pos_dist, dim=-1) + torch.std(neg_dist, dim=-1) + torch.mean(pos_dist, dim=-1) + torch.mean(F.relu(self.neg_thresh - neg_dist), dim=-1)
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
        # print(h_node_pert.shape, h_node_pert, poc_or_not)
        mask = poc_or_not == 1
        # print(selected_h_node_pert, 'selected_h_node_pert')
        h_node_pert = h_node_pert.to(torch.float16)
        time_embed_node = time_embed_node.to(torch.float16)
        poc_embed_node = poc_embed_node.to(torch.float16)
        h_node_pert = torch.cat([self.node_embedder(h_node_pert), time_embed_node, poc_embed_node], dim=-1)
        time_embed_edge = self.time_emb(t.index_select(0, batch_edge))
        h_edge_pert_float = h_edge_pert.float()
        h_edge_pert_embed = self.edge_embedder(h_edge_pert_float)
        h_edge_pert = torch.cat([h_edge_pert_embed, time_embed_edge], dim=-1)

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
        pred_node = h_node
        pred_halfedge = h_edge[:n_halfedges]+h_edge[n_halfedges:]
        pred_pos = pos_node
        
        return {
            'pred_node': pred_node,
            'pred_pos': pred_pos,
            'pred_halfedge': pred_halfedge,
        }  # at step 0

    @torch.no_grad()
    def sample(self, n_graphs, batch_node, halfedge_index, batch_halfedge, ref_data,
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

            # # use guidance to modify pos
            if guidance is not None:
                gui_type, gui_scale = guidance
                if (gui_scale > 0):
                    with torch.enable_grad():
                        h_node_in = h_node_pert.detach()
                        pos_in = pos_pert.detach().requires_grad_(True)
                        pred_bondpredictor = bond_predictor(h_node_in, pos_in, batch_node,
                                    edge_index, batch_edge, time_step)
                        if gui_type == 'entropy':
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'uncertainty':
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log().sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                        elif gui_type == 'uncertainty_bond':  # only for the predicted real bond (not no bond)
                            prob = torch.softmax(pred_bondpredictor, dim=-1)
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log()
                            uncertainty = (uncertainty * prob[:, 1:].detach().sum(dim=-1)).sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                        elif gui_type == 'entropy_bond':
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
                            entropy = entropy.log()
                            entropy = (entropy * prob_halfedge[:, 1:].detach().sum(dim=-1)).sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        else:
                            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
                    pos_prev = pos_prev + delta

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
        pred_node = torch.where(ref_data.atom_mask, node_init, node_traj[-1])
        pred_pos = torch.where(ref_data.pos_mask, pos_init, pos_traj[-1])
        pred_halfedge = torch.where(ref_data.bond_mask, halfedge_init, halfedge_traj[-1])


        return {
            'pred': [pred_node, pred_pos, pred_halfedge],
            'traj': [node_traj, pos_traj, halfedge_traj],
        }


