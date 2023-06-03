import os
import json
import pickle
import torch
from copy import deepcopy
from convlab2.util.file_util import read_zipped_json
import torch.utils.data as data

from convlab2.dpl.etc.util.state_structure import *
from convlab2.dpl.etc.util.vector_diachat import DiachatVector
from convlab2.dpl.etc.loader.build_data import build_data


class PolicyDataloader():
    def __init__(self):
        self.vector = DiachatVector()
        print('Start preprocessing the dataset')
        with open('convlab2/dpl/etc/data/complete_data.json', 'r') as f:
            source_data = json.load(f)
        self.data = source_data

    def create_dataset(self, part, batchsz, part_idx):
        print('Start creating {} dataset'.format(part))
        partdata = [self.data[i] for i in part_idx]
        targetdata = build_data(partdata)

        s = []
        a = []
        for item in targetdata:
            s.append(torch.Tensor(item[0]))
            a.append(torch.Tensor(item[1]))
        s = torch.stack(s)
        a = torch.stack(a)
        dataset = Dataset(s, a)
        dataloader = data.DataLoader(dataset, batchsz, True)
        return dataloader


class Dataset(data.Dataset):
    def __init__(self, s_s, a_s):
        self.s_s = s_s
        self.a_s = a_s
        self.num_total = len(s_s)

    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        return s, a

    def __len__(self):
        return self.num_total
