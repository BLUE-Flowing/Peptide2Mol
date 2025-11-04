import os
from typing import Optional

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import ConcatDataset, Dataset, random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data.components.mol_dataset import MolDataset
from src.data.components.collate import padded_collate_all


class MolDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "mol_data",
        batch_size: int = 4,
        infer_batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        data_seed: int = 42,
        reduction_factor: int = 1000,
        lmdb_fn: str = 'processed.lmdb',
        num_train: int = 100,

    ) -> None:
    
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.batch_size_per_device = self.hparams.batch_size
        self.infer_batch_size_per_device = self.hparams.infer_batch_size


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
            self.infer_batch_size_per_device = self.hparams.infer_batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        print(self.hparams,'self.hparams')
        # assert False
        self.all_data = MolDataset(self.hparams.data_dir, self.hparams.lmdb_fn)
        remaining = len(self.all_data) - self.hparams.num_train
        print(self.all_data, len(self.all_data))

        num_val = min(remaining, self.hparams.num_train // 5)

        if self.hparams.num_train + num_val < len(self.all_data):
            num_useless = len(self.all_data) - self.hparams.num_train - num_val
        else:
            num_useless = 0

        self.data_train, self.data_val, self.data_useless = random_split(
            self.all_data, [self.hparams.num_train, num_val, num_useless],
            generator=torch.Generator().manual_seed(self.hparams.data_seed)
        )
        print(f"Train: {len(self.data_train)}")
        print(f"Val: {len(self.data_val)}")

    def train_dataloader(self):
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                follow_batch = ['node_type', 'halfedge_type', 'diff_bond_type_idx', 'diffu_idx'],
                exclude_keys = ['orig_keys', 'pos_all_confs', 'smiles', 'num_confs', 'i_conf_listbond_index', 'bond_type', 'num_bonds', 'num_atoms'],
            )


    def val_dataloader(self):
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
                dataset=self.data_val,
                batch_size=self.infer_batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                follow_batch = ['node_type', 'halfedge_type', 'diff_bond_type_idx', 'diffu_idx'],
                exclude_keys = ['orig_keys', 'pos_all_confs', 'smiles', 'num_confs', 'i_conf_listbond_index', 'bond_type', 'num_bonds', 'num_atoms'],
                # collate_fn=padded_collate_all,
            )

    def test_dataloader(self):
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
                dataset=self.data_test,
                batch_size=self.infer_batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                follow_batch = ['node_type', 'halfedge_type', 'diff_bond_type_idx', 'diffu_idx'],
                exclude_keys = ['orig_keys', 'pos_all_confs', 'smiles', 'num_confs', 'i_conf_listbond_index', 'bond_type', 'num_bonds', 'num_atoms'],
            )
