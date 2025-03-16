from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import functional as F
from .mytransition import ContigousTransition, GeneralCategoricalTransition
from .mygraph import NodeEdgeNet
import torch_scatter
from .common import *
from .diffusion import *

def write_xyz_file(pred_pos, element, mask, filename="output_rep_0820_only_lig.xyz"):
    filtered_pos = pred_pos[~mask]
    filtered_element = element[~mask]
    
    n_atoms = filtered_pos.shape[0]  # Number of atoms after filtering
    with open(filename, 'w') as file:
        file.write(f"{n_atoms}\n")
        file.write(f"XYZ file generated from pred_pos and pred_node, index0\n")
        for j in range(n_atoms):
            x, y, z = filtered_pos[j]
            file.write(f"{filtered_element[j]} {x:.6f} {y:.6f} {z:.6f}\n")

def write_xyz_file_add(pred_pos, element, mask, filename="output_rep_0820_only_lig.xyz"):
    filtered_pos = pred_pos[~mask]
    filtered_element = element[~mask]
    
    n_atoms = filtered_pos.shape[0]  # Number of atoms after filtering
    with open(filename, 'a') as file:
        file.write(f"{n_atoms}\n")
        file.write(f"XYZ file generated from pred_pos and pred_node, index0\n")
        for j in range(n_atoms):
            x, y, z = filtered_pos[j]
            file.write(f"{filtered_element[j]} {x:.6f} {y:.6f} {z:.6f}\n")

class MolComp(Module):
    def __init__(self, kwargs):
        super().__init__()
        print(kwargs)
        self.num_node_types = kwargs['input_node_dim']
        self.num_edge_types = kwargs['input_edge_dim']
        self.bond_len_loss = kwargs['bond_len_loss']

        # # define beta and alpha
        self.define_betas_alphas(kwargs['diff'])

        self.node_embedder = nn.Linear(self.num_node_types, kwargs['node_dim_gui']-kwargs['time_dim_gui']-kwargs['poc_dim_gui'], bias=False)  # element type
        self.edge_embedder = nn.Linear(self.num_edge_types, kwargs['edge_dim_gui']-kwargs['time_dim_gui'], bias=False) # bond type
        self.time_emb = nn.Sequential(
            GaussianSmearing(stop=kwargs['diff']['num_timesteps'], num_gaussians=kwargs['time_dim'], type_='linear'),
        )
        self.poc_emb = nn.Embedding(2, kwargs['poc_dim_gui'])
        
        # # denoiser
        if kwargs['denoiser']['backbone'] == 'NodeEdgeNet':
            self.denoiser = NodeEdgeNet(kwargs['node_dim_gui'], kwargs['edge_dim_gui'], **kwargs['denoiser_gui'])
        else:
            raise NotImplementedError(kwargs['denoiser_gui']['backbone'])

        # # decoder
        self.node_decoder = MLP(kwargs['node_dim_gui'], self.num_node_types, kwargs['node_dim_gui'])
        self.edge_decoder = MLP(kwargs['edge_dim_gui'], self.num_edge_types, kwargs['edge_dim_gui'])


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

        # 3. Encode Positive Samples
        pos_embeddings = self.encode(
            pos_node_type, pos_node_pos, pos_batch_node,
            pos_halfedge_type, pos_halfedge_index, pos_batch_halfedge,
            time_step, poc_or_not[pos_mask], pos_diff_idx, pos_diff_pos_idx, pos_diff_bond_type_idx
        )

        # 4. Encode Negative Samples
        neg_embeddings = self.encode(
            neg_node_type, neg_node_pos, neg_batch_node,
            neg_halfedge_type, neg_halfedge_index, neg_batch_halfedge,
            time_step, poc_or_not[neg_mask], neg_diff_idx, neg_diff_pos_idx, neg_diff_bond_type_idx
        )

        # 5. Encode Original Samples (Anchor)
        anchor_embeddings = self.encode(
            node_type, node_pos, batch_node,
            halfedge_type, halfedge_index, batch_halfedge,
            time_step, poc_or_not, diff_idx, diff_pos_idx, diff_bond_type_idx
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
            pos_mask = rand_number < 0.5
            neg_mask = ~pos_mask
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
        node_idx = torch.arange(node_type.size(0), device=node_type.device)[mask]

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
        h_edge_pert = torch.cat([halfedge_type, halfedge_type], dim=0)

        # Forward pass through the model
        preds = self(
            h_node_pert, pos_pert, batch_node,
            h_edge_pert, edge_index, batch_edge, poc_or_not,
            time_step,
        )
        # Obtain embeddings (e.g., from the last layer or a pooling layer)
        embeddings = self.get_graph_embeddings(preds, batch_node)
        return embeddings

    def encode_nonoise(self, node_type, node_pos, batch_node,
            halfedge_type, halfedge_index, batch_halfedge,
            time_step, poc_or_not):

        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        h_edge_pert = torch.cat([halfedge_type, halfedge_type], dim=0)

        # Forward pass through the model
        preds = self.forward1(
            node_type, node_pos, batch_node,
            h_edge_pert, edge_index, batch_edge, poc_or_not,
            time_step,
        )
        return preds

    def encode_nonoise1(self, node_type, node_pos, batch_node,
            halfedge_type, halfedge_index, batch_halfedge,
            time_step, poc_or_not):

        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        h_edge_pert = torch.cat([halfedge_type, halfedge_type], dim=0)

        # Forward pass through the model
        preds = self(
            node_type, node_pos, batch_node,
            h_edge_pert, edge_index, batch_edge, poc_or_not,
            time_step,
        )
        embeddings = self.get_graph_embeddings(preds, batch_node)
        return embeddings.detach()


    def get_graph_embeddings(self, preds, batch_node):
        # Assuming preds['pred_node'] contains node features
        node_embeddings = preds['pred_node']
        # Aggregate node embeddings to graph embeddings (e.g., mean pooling)
        graph_embeddings = torch_scatter.scatter_mean(node_embeddings, batch_node, dim=0)
        return graph_embeddings


    def compute_contrastive_loss(self, anchor_embeddings, pos_embeddings, neg_embeddings):
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, dim=1)
        pos_embeddings = F.normalize(pos_embeddings, dim=1)
        neg_embeddings = F.normalize(neg_embeddings, dim=1)

        # Compute positive similarities
        pos_sim = torch.sum(anchor_embeddings * pos_embeddings, dim=1)

        # Compute negative similarities
        neg_sim = torch.sum(anchor_embeddings * neg_embeddings, dim=1)

        # Temperature parameter
        temperature = 300

        # Compute logits
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1) / temperature

        # Labels: positives are index 0
        labels = torch.zeros(anchor_embeddings.size(0), dtype=torch.long, device=anchor_embeddings.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss



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

    def forward1(self, 
                h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, poc_or_not, 
                t):
        """
        Predict Mol at step `0` given perturbed Mol at step `t` with hidden dims and time step
        """
        # 1 node and edge embedding + time embedding
        time_embed_node = self.time_emb(t.index_select(0, batch_node))
        poc_embed_node = self.poc_emb(poc_or_not)
        h_node_pert = h_node_pert.to(torch.float16)
        time_embed_node = time_embed_node.to(torch.float16)
        poc_embed_node = poc_embed_node.to(torch.float16)
        h_node_pert1 = h_node_pert.clone().detach()
        h_node_pert2 = torch.cat([self.node_embedder(h_node_pert1), time_embed_node, poc_embed_node], dim=-1)
        time_embed_edge = self.time_emb(t.index_select(0, batch_edge))
        h_edge_pert_float = h_edge_pert.float()
        h_edge_pert_embed = self.edge_embedder(h_edge_pert_float)
        h_edge_pert = torch.cat([h_edge_pert_embed, time_embed_edge], dim=-1)

        # 2 diffuse to get the updated node embedding and bond embedding
        h_node, pos_node, h_edge = self.denoiser(
            h_node=h_node_pert2,
            pos_node=pos_pert, 
            h_edge=h_edge_pert, 
            edge_index=edge_index,
            node_time=t.index_select(0, batch_node).unsqueeze(-1) / self.num_timesteps,
            edge_time=t.index_select(0, batch_edge).unsqueeze(-1) / self.num_timesteps,
        )
        
        n_halfedges = h_edge.shape[0] // 2
        pred_node = h_node
        
        return pred_node

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
        # print(batch_node.device, 'batch_node.device')
        time_embed_node = self.time_emb(t.index_select(0, batch_node))
        poc_embed_node = self.poc_emb(poc_or_not)
        # print(h_node_pert.shape, h_node_pert, poc_or_not)
        mask = poc_or_not == 1
        selected_h_node_pert = h_node_pert[mask]

        h_node_pert = h_node_pert.to(torch.float16)
        time_embed_node = time_embed_node.to(torch.float16)
        poc_embed_node = poc_embed_node.to(torch.float16)
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
        node_init1 = self.node_transition.sample_init(n_nodes_all)
        pos_init1 = self.pos_transition.sample_init([n_nodes_all, 3])
        halfedge_init1 = self.edge_transition.sample_init(n_halfedges_all)

        ref_data = ref_data.to(node_init1.device)
        node_init = torch.where(ref_data.atom_mask, ref_data.node_padding, node_init1)

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
        pred_bondpredictor_list = []
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
            
            if guidance is not None:
                gui_type, gui_scale = guidance
                if (gui_scale >= 0):
                    with torch.enable_grad():
                        h_node_in = h_node_pert
                        pos_in = pos_pert.detach().requires_grad_(True)
                        h_edge_in_half = h_halfedge_pert
                        # h_edge_in = torch.cat([h_edge_in_half, h_edge_in_half], dim=0)

                        pred_bondpredictor = bond_predictor.encode_nonoise(h_node_in, pos_in, batch_node,
                                                            h_edge_in_half, halfedge_index, batch_halfedge, time_step, ref_data.pocket_or_not, )
                        pred_bondpredictor1 = bond_predictor.encode_nonoise1(h_node_in, pos_in, batch_node,
                                                            h_edge_in_half, halfedge_index, batch_halfedge, time_step, ref_data.pocket_or_not, )
                       # pred_bondpredictor_list.append([pred_bondpredictor.detach()
                       # , pred_bondpredictor1.detach()
                       # ])
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


        pred_node = node_traj[-1][~ref_data.atom_mask].reshape(-1, ref_data.atom_mask.shape[-1])
        pred_pos = pos_traj[-1][~ref_data.pos_mask].reshape(-1, ref_data.pos_mask.shape[-1])
        pred_halfedge = torch.where(ref_data.bond_mask, halfedge_init, halfedge_traj[-1])
        origin_node = node_traj[-1][ref_data.atom_mask].reshape(-1, ref_data.atom_mask.shape[-1])
        origin_pos = pos_traj[-1][ref_data.pos_mask].reshape(-1, ref_data.pos_mask.shape[-1])
       # torch.save(pred_bondpredictor_list, traj_fn.replace('.xyz','_pred_bondpredictor_list.pt'))

        return {
            'pred': [pred_node, pred_pos, pred_halfedge],
            'traj': [node_traj, pos_traj, halfedge_traj],
            'origin': [origin_node, origin_pos, pred_halfedge],
        }
