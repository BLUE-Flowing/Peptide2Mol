from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.nn import functional as F
import pandas as pd
import numpy as np
from torch import ScriptModule, Tensor

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

    
    def model_step(self, batch, stage):
        assert self.losses is not None
        if stage == "train":
            node_for_loss, pred_node_for_loss, pos_for_loss, pred_pos_for_loss, halfedge_for_loss, pred_halfedge_for_loss = self.forward(batch)
            # print(torch.norm(pred_pos_for_loss))
            # print(pred_node_for_loss.shape, 'node')
            # print(pred_halfedge_for_loss.shape, 'halfedge')
            # exit()
            loss_pos = F.mse_loss(pred_pos_for_loss, pos_for_loss)
            loss_node = F.mse_loss(pred_node_for_loss, node_for_loss)  * 30
            loss_edge = F.mse_loss(pred_halfedge_for_loss, halfedge_for_loss) * 30
            loss_total = loss_pos + loss_node + loss_edge
            step_losses = loss_total
            step_losses_pos = loss_pos
            step_losses_node = loss_node
            step_losses_edge = loss_edge
            self.losses[stage]["loss"].append(step_losses.detach())
            self.losses[stage]["loss_pos"].append(step_losses_pos.detach())
            self.losses[stage]["loss_node"].append(step_losses_node.detach())
            self.losses[stage]["loss_edge"].append(step_losses_edge.detach())
        elif stage == "val":
            node_for_loss, pred_node_for_loss, pos_for_loss, pred_pos_for_loss, halfedge_for_loss, pred_halfedge_for_loss = self.forward(batch)
            loss_pos = F.mse_loss(pred_pos_for_loss, pos_for_loss)
            loss_node = F.mse_loss(pred_node_for_loss, node_for_loss)  * 30
            loss_edge = F.mse_loss(pred_halfedge_for_loss, halfedge_for_loss) * 30
            loss_total = loss_pos + loss_node + loss_edge
            step_losses = loss_total
            step_losses_pos = loss_pos
            step_losses_node = loss_node
            step_losses_edge = loss_edge
            self.losses[stage]["loss"].append(step_losses.detach())
            self.losses[stage]["loss_pos"].append(step_losses_pos.detach())
            self.losses[stage]["loss_node"].append(step_losses_node.detach())
            self.losses[stage]["loss_edge"].append(step_losses_edge.detach())
        elif stage == "test":
            # print(batch, 'batch in test')
            # exit()
            self.sample_diff(batch)
            step_losses = 0

        return step_losses

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # Combine the input tensors into a single tensor.
        try:
            loss = self.model_step(batch, "train")
            self.num_samples["train"] += len(batch)
            return loss
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in self.nn.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                print('cleaned')
                return None
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in self.nn.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return None
            else:
                #raise e
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
