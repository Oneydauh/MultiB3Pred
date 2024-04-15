import torch
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import pandas as pd
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import sys
from molecule_features import get_molecule_features
from utils import get_smile_feature
import pickle as pkl

class StandardData(Dataset):
    def __init__(self, data_dir,
                data_version='pei_cpp',
                mode='train', x_field = 'smiles', label_name = 'binary', train=False, ratio = None, seed=47, use_charge=True):
        # Set all input args as attributes
        self._indices=None
        self.transform = None
        self.label_name = label_name
        self.use_charge = use_charge

        file_names = data_version.split(';')
        esm = pkl.load(open(f'{data_dir}/{file_names[1]}.bin','rb'))
        a = esm['esm']
        data = pd.read_csv(f'{data_dir}/{file_names[0]}.csv',sep=',')
        b = data['smiles'].tolist()
        y = data['binary'].tolist()
        
        val = pkl.load(open(f'{data_dir}/D1valaug.bin','rb'))
        vala = val['esm']
        vald = pd.read_csv(f'{data_dir}/D1valaug.csv',sep=',')
        valb = vald['smiles'].tolist()
        valy = vald['binary'].tolist()
        
        if mode != 'dev':
            
            if train:
                self.a = a
                self.b = b
                self.y = y
            else:
                self.a = vala
                self.b = valb
                self.y = valy
            
        else:
            self.a = a
            self.b = b
            self.y = y
        self.__wrap_data__()

    def __wrapper__(self, esmfold, smiles, label):
        feature = get_smile_feature(smiles, use_charge = self.use_charge)
        feature = np.nan_to_num(feature)
        embedding, edge_attr, edge_index = get_molecule_features(smiles)
        embedding = embedding.numpy()
        edge_attr = edge_attr.numpy()

        esmfold = esmfold.clone().detach()

        if 'binary' in self.label_name:
            y = torch.tensor([label,], dtype=torch.long)
        else:
            y = torch.tensor([np.log(label),], dtype=torch.float32)

        return Data(x = torch.tensor(embedding, dtype=torch.float32), \
            y = y, \
            feature = torch.tensor(np.array(feature), dtype=torch.float32), \
            edge_attr= torch.tensor(edge_attr, dtype=torch.float32), \
            edge_index = torch.tensor(edge_index,dtype=torch.int64)[:,:2].t(), \
            esmfold = torch.tensor(esmfold, dtype=torch.float32)
        )

    def __wrap_data__(self):
        self.datas = [self.__wrapper__(esmfold, smiles, label) for esmfold, smiles, label in zip(self.a, self.b, self.y)]
    
    def len(self):
        return len(self.datas)

    def get(self, idx):
        return self.datas[idx]

