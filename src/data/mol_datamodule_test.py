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


class MolTestDataModule(LightningDataModule):

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

        self.data_test = MolDataset(self.hparams.data_dir, self.hparams.lmdb_fn)

        print(f"Test case number: {len(self.data_test)}")

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