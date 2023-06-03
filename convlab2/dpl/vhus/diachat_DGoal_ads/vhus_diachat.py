# -*- coding: utf-8 -*-
import os
import json
import torch
from convlab2.policy.vhus.diachat_DGoal_ads.goal_generator import GoalGenerator
from convlab2.policy.vhus.diachat_DGoal_ads.usermanager import UserDataManager
from convlab2.policy.vhus.diachat_DGoal_ads.usermodule import VHUS
from convlab2.policy.vhus.diachat_DGoal_ads.vhus import UserPolicyVHUSAbstract
from convlab2.policy.vhus.diachat_DGoal_ads.util import padding
from convlab2.policy.vhus.diachat_DGoal_ads.util import capital
from pprint import pprint

device = torch.device("cuda:0")


class UserPolicyVHUS(UserPolicyVHUSAbstract):
    def __init__(self, load_from_zip=False):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            config = json.load(f)
        manager = UserDataManager()
        voc_goal_size, voc_usr_size, voc_sys_size = manager.get_voc_size()
        self.user = VHUS(config, voc_goal_size, voc_usr_size,
                         voc_sys_size).to(device=device)
        self.goal_gen = GoalGenerator()
        self.manager = manager
        self.user.eval()

        if load_from_zip:
            self.user.load_state_dict(torch.load(
                'convlab2/policy/vhus/diachat_DGoal_ads/save/all_data_simulator.mdl'))
            self.user = self.user.cuda()

    def predict(self, sys_action=None):
        sys_action = sys_action
        # sys_action == None，代表用户需要说第一句话
        if sys_action == None:
            sys_seq_turn = ['<PAD>', '<PAD>', '<PAD>', '<PAD>']
        else:
            # sysda2seq中goal参数貌似没有用
            sys_seq_turn = self.manager.sysda2seq(
                self.manager.da_list_form_to_dict_form(sys_action), self.goal)

        self.sys_da_id_stack += self.manager.get_sysda_id([sys_seq_turn])
        sys_seq_len = torch.LongTensor(
            [max(len(sen), 1) for sen in self.sys_da_id_stack]).cuda(device=device)
        max_sen_len = sys_seq_len.max().item()
        sys_seq = torch.LongTensor(
            padding(self.sys_da_id_stack, max_sen_len)).cuda(device=device)
        usr_a, terminated = self.user.select_action(
            self.goal_input, self.goal_len_input, sys_seq, sys_seq_len)
        usr_action = self.manager.usrseq2da(
            self.manager.id2sentence(usr_a), self.goal)
        self.terminated = terminated
        # add value
        for i, u_ads in enumerate(usr_action):
            u_act, u_domain, u_slot = u_ads.split('-')
            if u_act == 'Inform':
                u_act = 'current'
            if u_act in ['AskFor', 'AskForSure', 'AskHow', 'AskWhy', 'Chitchat', 'current']:
                for j, (g_dsdone, g_value) in enumerate(self.goal[u_act]):
                    g_domain, g_slot, g_done = g_dsdone.split('-')
                    if g_domain == u_domain and g_slot == u_slot and g_done == 'False':
                        usr_action[i] = usr_action[i] + '-' + g_value
                        self.goal[u_act][j][0] = self.goal[u_act][j][0].replace('False', 'True')
                        break
            else:
                usr_action[i] = usr_action[i] + '-' + 'none'
        for i, u_ads in enumerate(usr_action):
            if len(u_ads.split('-')) == 3:
                usr_action[i] = usr_action[i] + '-' + 'NotInGoal'

        # update self.goal_input, self.goal_len_input for next turn
        self.goal_input = torch.LongTensor(self.manager.get_goal_id(self.manager.usrgoal2seq(self.goal))).cuda(device=device)
        self.goal_len_input = torch.LongTensor([len(self.goal_input)]).squeeze().cuda(device=device)
        
        """
        change usr_action form
            from: ['AskForSure-饮食-饮食名-枣', 'AskForSure-饮食-效果-升糖指数高']
            to: [['AskForSure', '饮食', '饮食名', '枣'], ['AskForSure', '饮食', '效果', '升糖指数高']]
        """
        dialog_act_tmp = []
        for adsv in usr_action:
            act, domain, slot, value = adsv.split('-')
            temp = [act, domain, slot, value]
            dialog_act_tmp.append(temp)
        usr_action = dialog_act_tmp
        return usr_action
