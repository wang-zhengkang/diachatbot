import os
import torch
import torch.utils.data as data

from convlab2.util.file_util import read_zipped_json
from convlab2.dpl.etc.util.state_structure import default_state
from convlab2.dpl.etc.util.vector_diachat import DiachatVector
from convlab2.dpl.etc.util.dst import RuleDST
from convlab2.dpl.etc.loader.build_data import build_data

class ActStateDataset(data.Dataset):
    def __init__(self, s_s, a_s, next_s):
        self.s_s = s_s
        self.a_s = a_s
        self.next_s = next_s
        self.num_total = len(s_s)
    
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        next_s = self.next_s[index]
        return s, a, next_s
    
    def __len__(self):
        return self.num_total

class EstimatorDataLoader():
    def __init__(self) -> None:
        self.vector = DiachatVector()
        self.data = {"train": [], "val": [], "test": []}
    def create_dataset_irl(self, part, batchsz):
        data_dir = "convlab2/dpl/etc/data"
        source_data = read_zipped_json(os.path.join(data_dir, f"{part}.json.zip"), f"{part}.json",)

        print(f'Start creating {part} dataset')

        self.data[part] = build_data(source_data)

        s = []
        a = []
        next_s = []
        for i, item in enumerate(self.data[part]):
            s.append(torch.Tensor(item[0]))
            a.append(torch.Tensor(item[1]))
            if item[0][-1]:  # terminated
                next_s.append(torch.Tensor(item[0]))
            else:
                next_s.append(torch.Tensor(self.data[part][i + 1][0]))
        s = torch.stack(s)
        a = torch.stack(a)
        next_s = torch.stack(next_s)
        dataset = ActStateDataset(s, a, next_s)
        dataloader = data.DataLoader(dataset, batchsz, True)
        print('Finish creating {} irl dataset'.format(part))
        return dataloader
