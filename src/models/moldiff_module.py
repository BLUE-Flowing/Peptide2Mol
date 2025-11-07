from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.nn import functional as F
import pandas as pd
import numpy as np
from torch import ScriptModule, Tensor
from torch.utils.data import DataLoader, Subset
from torch_scatter import scatter_mean

class MolLitModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        hparams: DictConfig = None,
    ) -> None:
        
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.nn = self.hparams.net

        self.losses = {}
        self.num_samples = {}
        self.results_dict = {}
        self.sample_losses = []
        self.sample_indices = []
        self.batch_count = 0
        self.max_batches = 5  # Configurable max batches for retraining trigger
        self.poor_case_threshold = 75  # Percentile for poor-performing cases
        self.extra_epochs = 3  # Number of extra epochs for retraining
        self.loss_filename = self.hparams.net.loss_filename
        self.is_extra_training = False  # Flag to track if we're in extra training
        with open(self.loss_filename, 'w') as f:
            f.write('name,total_loss,loss_pos,loss_node,loss_edge,batch_size,epoch,is_extra_training\n')

    def forward(self, x):
        # print(x)
        inputs = x
        return self.nn(inputs)
        
    def sample_diff(self, x):
        inputs = x
        return self.nn.sample(inputs)

    def on_validation_epoch_start(self) -> None:
        self._reset_losses_dict("val")
        self.results_dict = {}
        self.results_dict['id'] = []
        self.results_dict['outputs'] = []

    def on_train_epoch_start(self) -> None:
        self._reset_losses_dict("train")
        self.batch_count = 0
        self.sample_losses = []
        self.sample_indices = []


    def model_step(self, batch, stage, batch_idx=None, indices=None):
        assert self.losses is not None
        try:
            # Forward pass
            node_for_loss, pred_node_for_loss, pos_for_loss, pred_pos_for_loss, halfedge_for_loss, pred_halfedge_for_loss = self.forward(batch)

            # Log tensor shapes for debugging
            # if self.batch_count < self.max_batches:
            #     print(f"Batch {self.batch_count}, Stage: {stage}")
            #     print(f"node_for_loss shape: {node_for_loss.shape}, pred_node_for_loss shape: {pred_node_for_loss.shape}")
            #     print(f"pos_for_loss shape: {pos_for_loss.shape}, pred_pos_for_loss shape: {pred_pos_for_loss.shape}")
            #     print(f"halfedge_for_loss shape: {halfedge_for_loss.shape}, pred_halfedge_for_loss shape: {pred_halfedge_for_loss.shape}")

            # Compute per-sample losses
            # Ensure shapes match by checking dimensions
            # print(f"node_for_loss shape: {node_for_loss.shape}, pred_node_for_loss shape: {pred_node_for_loss.shape}")

            loss_pos = F.mse_loss(pred_pos_for_loss, pos_for_loss)  
            loss_node = F.mse_loss(pred_node_for_loss, node_for_loss) * 30  
            loss_edge = F.mse_loss(pred_halfedge_for_loss, halfedge_for_loss) * 30  
            # print(f"loss_pos: {loss_pos}, loss_node: {loss_node}, loss_edge: {loss_edge}")
            loss_total = loss_pos + loss_node + loss_edge  

            # Position loss
            loss_pos_per_element = F.mse_loss(pred_pos_for_loss, pos_for_loss, reduction='none')  # Shape: [409, 3]
            loss_pos_per_node = loss_pos_per_element.mean(dim=-1)  # Shape: [409]
            loss_pos1 = scatter_mean(loss_pos_per_node, batch.batch, dim=0, dim_size=8)  # Shape: [8]

            # Node loss
            loss_node_per_element = F.mse_loss(pred_node_for_loss, node_for_loss, reduction='none')  # Shape: [409, D_node]
            loss_node_per_node = loss_node_per_element.mean(dim=-1)  # Shape: [409]
            loss_node1 = scatter_mean(loss_node_per_node, batch.batch, dim=0, dim_size=8) * 30  # Shape: [8]

            # Edge loss
            loss_edge_per_element = F.mse_loss(pred_halfedge_for_loss, halfedge_for_loss, reduction='none')  # Shape: [27406, D_edge]
            loss_edge_per_edge = loss_edge_per_element.mean(dim=-1)  # Shape: [27406]
            loss_edge1 = scatter_mean(loss_edge_per_edge, batch.halfedge_type_batch, dim=0, dim_size=8) * 30  # Shape: [8]
            loss_total1 = loss_pos1 + loss_node1 + loss_edge1  # 形状: [batch_size]
            with open(self.loss_filename, 'a') as f:
                # f.write('name,total_loss,loss_pos,loss_node,loss_edge,batch_size,epoch\n')
                for i, name in enumerate(batch.name):
                    f.write(f'{name},{loss_total1[i].item():.6f},{loss_pos1[i].item():.6f},{loss_node1[i].item():.6f},{loss_edge1[i].item():.6f},{torch.max(batch.batch).item()+1},{self.current_epoch},{self.is_extra_training}\n')
            
            if stage == "train":
                step_losses = loss_total.mean()
                step_losses_pos = loss_pos.mean()
                step_losses_node = loss_node.mean()
                step_losses_edge = loss_edge.mean()
                self.losses[stage]["loss"].append(step_losses.detach())
                self.losses[stage]["loss_pos"].append(step_losses_pos.detach())
                self.losses[stage]["loss_node"].append(step_losses_node.detach())
                self.losses[stage]["loss_edge"].append(step_losses_edge.detach())
                # print(f"Step Loss {step_losses.item()}")
                # Track per-sample losses and indices for first 30 batches
                if self.batch_count < self.max_batches and indices is not None:
                    self.sample_losses.extend([loss_total1.detach().cpu().numpy()])
                    self.sample_indices.extend(indices.cpu().numpy())
            elif stage == "val":
                step_losses = loss_total.mean()
                step_losses_pos = loss_pos.mean()
                step_losses_node = loss_node.mean()
                step_losses_edge = loss_edge.mean()
                self.losses[stage]["loss"].append(step_losses.detach())
                self.losses[stage]["loss_pos"].append(step_losses_pos.detach())
                self.losses[stage]["loss_node"].append(step_losses_node.detach())
                self.losses[stage]["loss_edge"].append(step_losses_edge.detach())
            elif stage == "test":
                self.sample_diff(batch)
                step_losses = 0

            return step_losses

        except ValueError as e:
            print(f"Shape mismatch error: {e}")
            raise
        except RuntimeError as e:
            print(f"Runtime error in model_step: {e}")
            raise

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        try:
            # Extract indices from batch, assuming they are included
            if isinstance(batch, tuple) and len(batch) > 2:
                indices = batch[-1]  # Adjust based on your batch structure
            else:
                # Fallback: Generate indices based on batch_idx and batch size
                batch_size = len(batch[0]) if isinstance(batch, tuple) else len(batch)
                indices = torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size, device=self.device)
            
            loss = self.model_step(batch, "train", batch_idx, indices)
            self.num_samples["train"] += len(batch[0]) if isinstance(batch, tuple) else len(batch)
            self.batch_count += 1

            # Trigger retraining after 30 batches
            if self.batch_count == self.max_batches and self.sample_losses:
                self._retrain_poor_cases()

            return loss
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in self.nn.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
                print('cleaned')
                return None
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in self.nn.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
                return None
            else:
                print(e)
                return None

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        try:
            loss = self.model_step(batch, "val")
            self.num_samples["val"] += len(batch)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in self.nn.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                # continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in self.nn.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                # continue
            else:
                #raise e
                print(e)
                # continue

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if not self.trainer.sanity_checking:
            result_dict = {
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            }
            result_dict.update(self._get_mean_loss_dict_for_type("train"))
            result_dict.update(self._get_mean_loss_dict_for_type("val"))
            self.log_dict(result_dict, sync_dist=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.model_step(batch, "test")
        # self.num_samples["test"] += len(batch)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.nn = torch.compile(self.nn)

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.hparams.lr
        super().optimizer_step(*args, **kwargs)

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        """Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your own
        implementation if you need to.

        Args:
            loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
                holds the normalized value (scaled by 1 / accumulation steps).

        Example::

            def backward(self, loss):
                loss.backward()

        """
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            try:
                loss.backward(*args, **kwargs)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    for p in self.nn.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    # continue
                elif 'Input mismatch' in str(e):
                    print('| WARNING: weird torch_cluster error, skipping batch')
                    for p in self.nn.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    # continue
                else:
                    #raise e
                    print(e)
                    # continue


    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "train_loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            else:
                return {
                    "optimizer": optimizer, 
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                    },
                }
        return {"optimizer": optimizer}
    
    def _get_mean_loss_dict_for_type(self, stage):
        assert self.losses is not None
        mean_losses = {}
        # print(self.losses,'self.losses')
        for loss_fn_name in self.losses[stage].keys():
            mean_losses[stage + "_" + loss_fn_name] = torch.stack(
                self.losses[stage][loss_fn_name]
            ).sum() / self.num_samples[stage]
        # print(mean_losses, 'mean losses')
        return mean_losses
    
    def _reset_losses_dict(self, stage):
        self.losses[stage] = defaultdict(list)
        self.num_samples[stage] = 0

    def _retrain_poor_cases(self):
        # Identify poor-performing cases
        sample_losses = np.array(self.sample_losses).flatten()
        sample_indices = np.array(self.sample_indices)
        threshold = np.percentile(sample_losses, self.poor_case_threshold)
        poor_indices = sample_indices[sample_losses > threshold]

        if len(poor_indices) == 0:
            print('No poor-performing cases found. Continuing training.')
            # Reset tracking for the next 30 batches
            self.sample_losses = []
            self.sample_indices = []
            self.batch_count = 0
            return

        # Create a subset dataset for poor-performing cases
        poor_dataset = Subset(self.trainer.train_dataloader.dataset, poor_indices)
        
        # Use the same collate function as the original dataloader
        poor_dataloader = DataLoader(
            poor_dataset,
            batch_size=self.trainer.train_dataloader.batch_size//2,
            shuffle=True,
            num_workers=self.trainer.train_dataloader.num_workers,
            pin_memory=True,
            collate_fn=self.trainer.train_dataloader.collate_fn
        )

        # Train poor-performing cases for extra epochs
        self.is_extra_training = True  # Set flag for extra training
        for epoch in range(self.extra_epochs):
            self._reset_losses_dict("train")
            self.num_samples["train"] = 0
            for batch_idx, batch in enumerate(poor_dataloader):
                # Move batch to GPU
                if isinstance(batch, tuple):
                    batch = tuple(b.to(self.device) for b in batch)
                else:
                    batch = batch.to(self.device)
                    
                loss = self.training_step(batch, batch_idx)
                if loss is not None:
                    # Backward pass
                    self.backward(loss)
        
        self.is_extra_training = False  # Reset flag after extra training
        # Reset tracking for the next 30 batches
        self.sample_losses = []
        self.sample_indices = []
        self.batch_count = 0
