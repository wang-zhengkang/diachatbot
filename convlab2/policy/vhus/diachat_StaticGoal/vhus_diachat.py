# -*- coding: utf-8 -*-
import os
import json
import torch
from convlab2.task.diachat.goal_generator2 import GoalGenerator
from convlab2.policy.vhus.diachat_StaticGoal.usermanager import UserDataManager
from convlab2.policy.vhus.diachat_StaticGoal.usermodule import VHUS
from convlab2.policy.vhus.diachat_StaticGoal.vhus import UserPolicyVHUSAbstract
from convlab2.policy.vhus.diachat_StaticGoal.util import padding
from convlab2.policy.vhus.diachat_StaticGoal.util import capital

device = torch.device("cuda:0")

DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "best_simulator.mdl")


class UserPolicyVHUS(UserPolicyVHUSAbstract):
    def __init__(self,
                 load_from_zip=False,
                 archive_file=DEFAULT_ARCHIVE_FILE,
                 model_file=''):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            config = json.load(f)
        manager = UserDataManager()
        voc_goal_size, voc_usr_size, voc_sys_size = manager.get_voc_size()
        self.user = VHUS(config, voc_goal_size, voc_usr_size, voc_sys_size).to(device=device)
        self.goal_gen = GoalGenerator()
        self.manager = manager
        self.user.eval()

        if load_from_zip:
            self.user.load_state_dict(torch.load('convlab2/policy/vhus/diachat_StaticGoal/save/all_data_simulator.mdl'))
            self.user = self.user.cuda()

    def predict(self, state=[['<PAD>', '<PAD>', '<PAD>', '<PAD>']]):
        sys_action = state

        sys_seq_turn = self.manager.sysda2seq(self.manager.da_list_form_to_dict_form(sys_action), self.goal)
        self.sys_da_id_stack += self.manager.get_sysda_id([sys_seq_turn])
        sys_seq_len = torch.LongTensor([max(len(sen), 1) for sen in self.sys_da_id_stack]).cuda(device=device)
        max_sen_len = sys_seq_len.max().item()
        sys_seq = torch.LongTensor(padding(self.sys_da_id_stack, max_sen_len)).cuda(device=device)
        usr_a, terminated = self.user.select_action(self.goal_input, self.goal_len_input, sys_seq, sys_seq_len)
        usr_action = self.manager.usrseq2da(self.manager.id2sentence(usr_a), self.goal)
        self.terminated = terminated
        # return usr_action


        # usr_action = self.manager.da_dict_form_to_list_form(usr_action)
        # add value
        # tag = 0
        # for ads in usr_action:
        #     split_ads = ads.split('-')
        #     if split_ads[0] in ['Accept', 'AskWhy', 'Assure', 'Deny', 'Chitchat']:
        #         usr_action[tag] += '-none'
        #     else:
        #         try:
        #             exec(f"value = next(self.{split_ads[0]}_{split_ads[1]}_{split_ads[2]})")
        #             print("exec执行成功")
        #         except:
        #             value = 'none'
        #         usr_action[tag] += f'-{value}'
        #     tag += 1

        return usr_action

