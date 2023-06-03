import os
import json
import pickle
import torch
import numpy as np
import torch.utils.data as data
from convlab2.policy.mle.diachat_decouple.util.dst import RuleDST
from copy import deepcopy


class PolicyDataLoaderDiachat():

    def __init__(self, vectoriser):
        self.vector = vectoriser
        
        print('Start preprocessing the dataset')
        self._build_data()

    def _build_data(self):
        with open('data/diachat/annotations_goal.json', 'r') as f:
            source_data = json.load(f)
        self.data = []
        dst = RuleDST()

        for conversation in source_data:
            sess = conversation['utterances']
            dst.init_session()
            for i, turn in enumerate(sess):
                domains = turn['domain']
                if domains == '':
                    domains = 'none'

                sys_action = []  # 系统动作 作为训练数据的Y

                if turn['agentRole'] == 'User':
                    dst.state['cur_domain'] = domains
                    usr_action = []
                    for da in turn['annotation']:
                        act = da['act_label']
                        for anno in da['slot_values']:
                            adsv = []
                            adsv.append(act)
                            adsv.append(anno['domain'])
                            adsv.append(anno['slot'])
                            adsv.append(anno['value'])
                            usr_action.append(adsv)
                    dst.state['usr_action'] = usr_action

                    if i + 2 == len(sess):
                        dst.state['terminated'] = True

                else:
                    dst.state['belief_state'] = turn['sys_state_init']

                    sys_action = []
                    for da in turn['annotation']:
                        act = da['act_label']
                        for anno in da['slot_values']:
                            adsv = []
                            adsv.append(act)
                            adsv.append(anno['domain'])
                            adsv.append(anno['slot'])
                            adsv.append(anno['value'])
                            sys_action.append(adsv)

                    training_X = self.vector.state_vectorize(deepcopy(dst.state))
                    training_Y = self.vector.action_vectorize(sys_action)
                    self.data.append([training_X, training_Y])
                    dst.state['sys_action'] = sys_action
        pass
        

    def _load_data(self, processed_dir):
        self.data = {}
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'rb') as f:
                self.data[part] = pickle.load(f)

    def create_dataset(self, part, batchsz, part_idx):
        print('Start creating {} dataset'.format(part))
        partdata = [self.data[i]for i in part_idx]
        s = []
        a = []
        for item in partdata:
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

