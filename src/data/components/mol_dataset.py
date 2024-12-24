import bisect
import pickle
from pathlib import Path
import lmdb, os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from src.data.utils.mytransformsnb import FeaturizeMol, Compose
from torch_geometric.data import Data

class Drug3DData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_drug3d_dicts(ligand_dict=None, **kwargs):
        instance = Drug3DData(**kwargs)

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance[key] = item
            instance['orig_keys'] = list(ligand_dict.keys())

        # instance['nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'bond_index':
            return len(self['node_type'])
        elif key == 'edge_index':
            return len(self['node_type'])
        elif key == 'halfedge_index':
            return len(self['node_type'])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class MolDataset(Dataset):

    def __init__(self, root, lmdb_fn, transform=None):
        super().__init__()
        self.root = root
        
        self.processed_path = os.path.join(root, lmdb_fn)
        # self.filter = filter

        self.transform = Compose([FeaturizeMol()])
        self.db = None
        self.keys = None

        if (not os.path.exists(self.processed_path)):
            print('Data need to be preprocessed firstly!')
            print(self.processed_path, os.path.exists(self.processed_path), os.getcwd())
            exit()

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)
    
    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        # data.id = idx
        if self.transform is not None:
            data = self.transform(data)
        return data
 
