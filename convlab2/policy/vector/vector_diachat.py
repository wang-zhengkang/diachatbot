import os
import json
import numpy as np
from convlab2.policy.vec import Vector
from convlab2.util.diachat.state_structure import default_state
from convlab2.util.diachat.state_structure import belief_state_vectorize
from convlab2.util.diachat.domain_act_slot import *
from convlab2.util.diachat.lexicalize import delexicalize_da, lexicalize_da


# from convlab2.util.diachat.dbquery import Database


class DiachatVector(Vector):
    def __init__(self, sys_domain_da_voc_json, usr_domain_da_voc_json, sys_da_voc_json, usr_da_voc_json):
        
        # self.sys_domain_da_voc为ad_voc不含slot
        self.sys_domain_da_voc = json.load(open(sys_domain_da_voc_json, encoding='utf-8'))
        self.usr_domain_da_voc = json.load(open(usr_domain_da_voc_json, encoding='utf-8'))

        # self.sys_da_voc为ads_voc包含slot
        self.sys_da_voc = json.load(open(sys_da_voc_json, encoding='utf-8'))
        self.usr_da_voc = json.load(open(usr_da_voc_json, encoding='utf-8'))
        
        # self.sys_intents = ["Advice", "AdviceNot", "AskFor", "AskForSure", "AskHow", "Assure", "Chitchat", "Deny","Explanation", "GeneralExplanation", "GeneralAdvice"]
        # self.user_intents = ["Accept", "AskFor", "AskForSure", "AskHow", "AskWhy", "Assure", "Chitchat", "Inform","Uncertain"]
        self.sys_intents = act_labels['doctor_act']
        self.user_intents = act_labels['user_act']

        # self.all_domains = ['基本信息','行为','治疗','问题','运动','饮食']
        self.all_domains = list(domain_slots.keys())
        self.all_domains.append('none')

        self.generate_dict()

    def generate_dict(self):

        self.sys_da2id = dict((a, i) for i, a in enumerate(self.sys_da_voc))
        self.id2sys_da = dict((i, a) for i, a in enumerate(self.sys_da_voc))
        self.sys_da_dim = len(self.sys_da_voc)

        self.usr_da2id = dict((a, i) for i, a in enumerate(self.usr_da_voc))
        self.id2usr_da = dict((i, a) for i, a in enumerate(self.usr_da_voc))
        self.usr_da_dim = len(self.usr_da_voc)

        self.sys_domain_da2id = dict((a, i) for i, a in enumerate(self.sys_domain_da_voc))
        self.id2sys_domain_da = dict((i, a) for i, a in enumerate(self.sys_domain_da_voc))
        self.sys_domain_da_dim = len(self.sys_domain_da_voc)

        self.usr_domain_da2id = dict((a, i) for i, a in enumerate(self.usr_domain_da_voc))
        self.id2usr_domain_da = dict((i, a) for i, a in enumerate(self.usr_domain_da_voc))
        self.usr_domain_da_dim = len(self.usr_domain_da_voc)

        self.sys_intents2id = dict((a, i) for i, a in enumerate(self.sys_intents))
        self.id2sys_intents = dict((i, a) for i, a in enumerate(self.sys_intents))
        self.sys_intents_dim = len(self.sys_intents)

        self.usr_intents2id = dict((a, i) for i, a in enumerate(self.user_intents))
        self.id2usr_intents = dict((i, a) for i, a in enumerate(self.user_intents))
        self.usr_intents_dim = len(self.user_intents)

        self.domain2id = dict((a, i) for i, a in enumerate(self.all_domains))
        self.id2domain = dict((i, a) for i, a in enumerate(self.all_domains))
        self.domain_dim = len(self.all_domains)

        self.belief_state_domainslot2id = dict()  # 没啥用
        self.belief_state_id2domainslot = dict()  # 没啥用
        self.belief_state_dim = 0
        for domain in domain_slots.keys():
            if domain == '基本信息':
                self.belief_state_domainslot2id[domain] = domain_slots2id[domain]
                self.belief_state_id2domainslot[domain] = id2domain_slots[domain]
                self.belief_state_dim += len(domain_slots2id[domain])
            else:
                self.belief_state_domainslot2id[domain] = {'现状': domain_slots2id[domain],
                                                           '解释': domain_slots2id[domain]}
                self.belief_state_id2domainslot[domain] = {'现状': id2domain_slots[domain],
                                                           '解释': id2domain_slots[domain]}
                self.belief_state_dim += 2 * len(domain_slots2id[domain])
                if domain != '问题':
                    self.belief_state_domainslot2id[domain]['建议'] = domain_slots2id[domain]
                    self.belief_state_id2domainslot[domain]['建议'] = id2domain_slots[domain]
                    self.belief_state_dim += len(domain_slots2id[domain])

        # action维度
        self.action_dim = self.sys_da_dim + \
                          self.usr_da_dim + \
                          self.sys_domain_da_dim + \
                          self.usr_domain_da_dim + \
                          self.sys_intents_dim + \
                          self.usr_intents_dim

        # state维度=6+442+104=553
        self.state_dim = self.domain_dim + self.action_dim + self.belief_state_dim

    def state_vectorize(self, state):
        self.belief_state = state['belief_state']  # 这里是顾客已知部分，变成26宽度
        self.cur_domain = state['cur_domain']

        da = state['usr_action']  # da必定为空，这里有点奇怪
        de_act, de_domain, de_act_domain, da = delexicalize_da(da)  # 改成带减号如dec的那种
        cur_domain_vec = np.zeros(self.domain_dim)
        for a in de_domain:
            if a in self.domain2id:
                cur_domain_vec[self.domain2id[a]] = 1
        usr_intent_vec = np.zeros(self.usr_intents_dim)
        for a in de_act:
            if a in self.usr_intents2id:
                usr_intent_vec[self.usr_intents2id[a]] = 1
        usr_act_domain_vec = np.zeros(self.usr_domain_da_dim)
        for a in de_act_domain:
            if a in self.usr_domain_da2id:
                usr_act_domain_vec[self.usr_domain_da2id[a]] = 1
        usr_act_vec = np.zeros(self.usr_da_dim)
        for a in da:
            if a in self.usr_da2id:
                usr_act_vec[self.usr_da2id[a]] = 1.  # 把这一部分矩阵化

        da = state['sys_action']
        de_act, de_domain, de_act_domain, da = delexicalize_da(da)  # 改成带减号如dec的那种
        sys_intent_vec = np.zeros(self.sys_intents_dim)
        for a in de_act:
            if a in self.sys_intents2id:
                sys_intent_vec[self.sys_intents2id[a]] = 1
        sys_act_domain_vec = np.zeros(self.sys_domain_da_dim)
        for a in de_act_domain:
            if a in self.sys_domain_da2id:
                sys_act_domain_vec[self.sys_domain_da2id[a]] = 1
        sys_act_vec = np.zeros(self.sys_da_dim)
        for a in da:
            if a in self.sys_da2id:
                sys_act_vec[self.sys_da2id[a]] = 1.  # 把这一部分矩阵化

        # 把belief矩阵化
        belief_state_vec = belief_state_vectorize(state['belief_state'])

        #         sys_domain_vec = np.zeros(self.sys_domain_da_dim)
        #         sys_domain = state['sys_domain']
        #         if sys_domain in self.sys_domain_da2id:
        #             sys_domain_vec[self.sys_domain_da2id[sys_domain]] = 1.
        #
        #         user_domain_vec = np.zeros(self.usr_domain_da_dim)
        #         user_domain = state['user_domain']
        #         if user_domain in self.usr_domain_da2id:
        #                 user_domain_vec[self.usr_domain_da2id[user_domain]] = 1.
        #
        #         terminated = 1. if state['terminated'] else 0.

        # state矩阵
        state_vec = np.r_[cur_domain_vec,  # 当前领域
        usr_intent_vec,  # usr act_label
        usr_act_domain_vec,  # usr act_label+domain
        usr_act_vec,  # usr act_label+domain+slot
        sys_intent_vec,  # sys act_label
        sys_act_domain_vec,  # sys act_label+domain
        sys_act_vec,  # sys act_label+domain+slot
        belief_state_vec]  # belief state

        return state_vec  # 返回状态矩阵

    def action_devectorize(self, action_vec):
        """
        must call state_vectorize func before
        :param action_vec:
        :return:
        """
        da = []
        for i, idx in enumerate(action_vec):
            if idx == 1:
                da.append(self.id2sys_da[i])  # 转化为形如['Inform+餐馆+名称+1']

        return da

    def action_vectorize(self, da):
        _, _, _, da1 = delexicalize_da(da)
        sys_act_vec = np.zeros(self.sys_da_dim)
        for a in da1:
            if a in self.sys_da2id:
                sys_act_vec[self.sys_da2id[a]] = 1.  # 把这一部分矩阵化
        return sys_act_vec


#         belief_state_vec = np.zeros(self.belief_state_dim)
#         for a in da1:
#             intent, domain, slot = a.split('+')
#             domain_slot=domain+'-'+slot
#             if domain_slot in self.belief_state_da2id:
#                 i=self.belief_state_da2id[domain_slot]
#                 belief_state_vec[i]=1
#  
#         da2 = delexicalize_da(da)
#         sys_act_vec = np.zeros(self.sys_da_dim)
#         for a in da2:
#             if a in self.sys_da2id:
#                 sys_act_vec[self.sys_da2id[a]] = 1.
#  
#         sys_domain_vec = np.zeros(self.domain_dim)
#         if sys_domains is not 'none':
#             domains = sys_domains.split('-')
#             for domain in domains:
#                 if domain in self.domain2id:
#                     sys_domain_vec[self.domain2id[domain]] = 1.
#  
#         sys_intents_vec = np.zeros(self.sys_intents_dim)
#         for sys_intent in sys_intents:
#             if sys_intent in self.sys_intents2id:
#                 sys_intents_vec[self.sys_intents2id[sys_intent]] = 1. # 把这一部分矩阵化
#  
#         action_vec = np.r_[sys_act_vec, sys_domain_vec,sys_intents_vec]
#  
#         return action_vec


if __name__ == '__main__':
    vec = DiachatVector('../../../data/diachat/sys_act_domain.json',
                        '../../../data/diachat/usr_act_domain.json',
                        '../../../data/diachat/sys_act_domain_slot.json',
                        '../../../data/diachat/usr_act_domain_slot.json')

    print('state_dim', vec.state_dim)
    print('action_dim', vec.sys_da_dim)
    print('sys_intents_dim', vec.sys_intents_dim)
    print('domain_dim', vec.domain_dim)
    print('sys_da_dim', vec.sys_da_dim)
    print('usr_da_dim', vec.usr_da_dim)
    print('belief_state_dim', vec.belief_state_dim)
