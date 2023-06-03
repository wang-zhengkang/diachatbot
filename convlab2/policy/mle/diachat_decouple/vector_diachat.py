import json
import numpy as np
from convlab2.dpl.vec import Vector
from convlab2.dpl.etc.util.domain_act_slot import *
from convlab2.dpl.etc.util.dexicalize import delexicalize_da


class DiachatVector(Vector):

    def __init__(self, sys_ads_file, usr_ads_file, sys_ad_file, usr_ad_file):
    
        # ads: act-domain-slot  ad: act-domain
        self.sys_ads_voc = json.load(open(sys_ads_file, encoding='UTF-8'))
        self.usr_ads_voc = json.load(open(usr_ads_file, encoding='UTF-8'))
        self.sys_ad_voc = json.load(open(sys_ad_file, encoding='UTF-8'))
        self.usr_ad_voc = json.load(open(usr_ad_file, encoding='UTF-8'))
        
        self.sys_act = sys_act
        self.usr_act = usr_act

        self.all_domain = domain
        
        self.generate_dict()

    def generate_dict(self):

        self.sys_ads2id = dict((a, i) for i, a in enumerate(self.sys_ads_voc))
        self.id2sys_ads = dict((i, a) for i, a in enumerate(self.sys_ads_voc))
        self.sys_ads_dim = len(self.sys_ads_voc)

        self.usr_ads2id = dict((a, i) for i, a in enumerate(self.usr_ads_voc))
        self.id2usr_ads = dict((i, a) for i, a in enumerate(self.usr_ads_voc))
        self.usr_ads_dim = len(self.usr_ads_voc)

        self.sys_ad2id = dict((a, i) for i, a in enumerate(self.sys_ad_voc))
        self.id2sys_ad = dict((i, a) for i, a in enumerate(self.sys_ad_voc))
        self.sys_ad_dim = len(self.sys_ad_voc)

        self.usr_ad2id = dict((a, i) for i, a in enumerate(self.usr_ad_voc))
        self.id2usr_ad = dict((i, a) for i, a in enumerate(self.usr_ad_voc))
        self.usr_ad_dim = len(self.usr_ad_voc)

        self.sys_act2id = dict((a, i) for i, a in enumerate(self.sys_act))
        self.id2sys_act = dict((i, a) for i, a in enumerate(self.sys_act))
        self.sys_act_dim = len(self.sys_act)

        self.usr_act2id = dict((a, i) for i, a in enumerate(self.usr_act))
        self.id2usr_act = dict((i, a) for i, a in enumerate(self.usr_act))
        self.usr_act_dim = len(self.usr_act)

        self.domain2id = dict((a, i) for i, a in enumerate(self.all_domain))
        self.id2domain = dict((i, a) for i, a in enumerate(self.all_domain))
        self.domain_dim = len(self.all_domain)

        self.belief_state_domainslot2id = dict()  # 没啥用
        self.belief_state_id2domainslot = dict()  # 没啥用
        self.belief_state_dim = 0

        self.domain_slot2id = dict()
        self.id2domain_slot = dict()
        for d, slots in domain_slot.items():
            self.domain_slot2id[d] = dict((slots[i], i) for i in range(len(slots)))
            self.id2domain_slot[d] = dict((i, slots[i]) for i in range(len(slots)))
            
        for domain in all_domain:
            if domain == '基本信息':
                self.belief_state_domainslot2id[domain] = self.domain_slot2id[domain]
                self.belief_state_id2domainslot[domain] = self.id2domain_slot[domain]
                self.belief_state_dim += len(self.domain_slot2id[domain])
            else:
                self.belief_state_domainslot2id[domain] = {'现状': self.domain_slot2id[domain],
                                                           '解释': self.domain_slot2id[domain]}
                self.belief_state_id2domainslot[domain] = {'现状': self.id2domain_slot[domain],
                                                           '解释': self.id2domain_slot[domain]}
                self.belief_state_dim += 2 * len(self.domain_slot2id[domain])
                if domain != '问题':
                    self.belief_state_domainslot2id[domain]['建议'] = self.domain_slot2id[domain]
                    self.belief_state_id2domainslot[domain]['建议'] = self.id2domain_slot[domain]
                    self.belief_state_dim += len(self.domain_slot2id[domain])

        # self.action_dim = self.sys_ads_dim + \
        #                   self.usr_ads_dim + \
        #                   self.sys_ad_dim + \
        #                   self.usr_ad_dim + \
        #                   self.sys_act_dim + \
        #                   self.usr_act_dim
        # self.action_dim = self.usr_ad_dim + self.usr_act_dim

        # self.state_dim = self.domain_dim + self.action_dim + self.belief_state_dim
        self.state_dim = self.domain_dim + self.belief_state_dim

    @staticmethod
    def belief_state_vectorize(self, belief_state):
        def domain_group_vectorize(belief_state, domain, group):
            has_slot_vec = np.zeros(len(self.domain_slot2id[domain]))
            if domain in belief_state:
                domain_belief = belief_state[domain]
                if group in domain_belief:
                    slot_values_group = domain_belief[group]
                    slots_with_value = []
                    for svs in slot_values_group:
                        for s, v in svs.items():
                            if v != '' and (s not in ['是否建议', '状态']):  # 临时处理一下这两个
                                slots_with_value.append(s)
                    slots_with_value = list(set(slots_with_value))
                    for slot in slots_with_value:
                        has_slot_vec[self.domain_slot2id[domain][slot]] = 1
            return has_slot_vec
        belief_state_vector = []
        for domain in all_domain:
            if domain == '基本信息':
                has_slot_vec = domain_group_vectorize(belief_state, domain, '现状')
                belief_state_vector = np.hstack((belief_state_vector, has_slot_vec))
            else:
                has_slot_vec = domain_group_vectorize(belief_state, domain, '现状')
                belief_state_vector = np.hstack((belief_state_vector, has_slot_vec))

                has_slot_vec = domain_group_vectorize(belief_state, domain, '解释')
                belief_state_vector = np.hstack((belief_state_vector, has_slot_vec))

                if domain != '问题':
                    has_slot_vec = domain_group_vectorize(belief_state, domain, '建议')
                    belief_state_vector = np.hstack((belief_state_vector, has_slot_vec))

        return belief_state_vector
    
    def state_vectorize(self, state):
        self.belief_state = state['belief_state']
        self.cur_domain = state['cur_domain']

        da = state['usr_action']

        # da = [['AskForSure', '问题', '症状', '眼皮肿'],
        #       ['AskForSure', '问题', '症状', '血糖高'],
        #       ['Inform', '问题', '血糖值', '挺好的'],
        #       ['Inform', '问题', '时间', '这几天']]

        # de_act = ['AskForSure', 'Inform']
        # de_domain = ['问题']
        # de_act_domain = ['Inform-问题', 'AskForSure-问题']
        # da = ['Inform-问题-血糖值', 'Inform-问题-时间', 'AskForSure-问题-症状']
        de_act, de_domain, de_act_domain, da = delexicalize_da(da)  # 改成带减号如dec的那种
        cur_domain_vec = np.zeros(self.domain_dim)
        for a in de_domain:
            if a in self.domain2id:
                cur_domain_vec[self.domain2id[a]] = 1
        usr_act_vec = np.zeros(self.usr_act_dim)
        for a in de_act:
            if a in self.usr_act2id:
                usr_act_vec[self.usr_act2id[a]] = 1
        usr_ad_vec = np.zeros(self.usr_ad_dim)
        for a in de_act_domain:
            if a in self.usr_ad2id:
                usr_ad_vec[self.usr_ad2id[a]] = 1
        usr_ads_vec = np.zeros(self.usr_ads_dim)
        for a in da:
            if a in self.usr_ads2id:
                usr_ads_vec[self.usr_ads2id[a]] = 1

        da = state['sys_action']
        de_act, de_domain, de_act_domain, da = delexicalize_da(da)  # 改成带减号如dec的那种
        sys_act_vec = np.zeros(self.sys_act_dim)
        for a in de_act:
            if a in self.sys_act2id:
                sys_act_vec[self.sys_act2id[a]] = 1
        sys_ad_vec = np.zeros(self.sys_ad_dim)
        for a in de_act_domain:
            if a in self.sys_ad2id:
                sys_ad_vec[self.sys_ad2id[a]] = 1
        sys_ads_vec = np.zeros(self.sys_ads_dim)
        for a in da:
            if a in self.sys_ads2id:
                sys_ads_vec[self.sys_ads2id[a]] = 1

        belief_state_vec = self.belief_state_vectorize(self, state['belief_state'])

        # state_vec = np.r_[
        #     cur_domain_vec,  # 6
        #     usr_act_vec,  # 9
        #     usr_ad_vec,  # 33
        #     usr_ads_vec,  # 152
        #     sys_act_vec,  # 11
        #     sys_ad_vec,  # 50
        #     sys_ads_vec,  # 186
        #     belief_state_vec  # 104
        # ]
        state_vec = np.r_[
            cur_domain_vec,  # 6
            belief_state_vec  # 104
        ]
        return state_vec  # state_vec dim: 551

    def action_devectorize(self, action_vec):
        """
        must call state_vectorize func before
        :param action_vec:
        :return:
        """
        da = []
        for i, idx in enumerate(action_vec):
            if idx == 1:
                da.append(self.id2sys_ads[i])  # 转化为形如['Inform+餐馆+名称+1']

        return da

    def action_vectorize(self, da):
        _, _, _, da1 = delexicalize_da(da)
        sys_ads_vec = np.zeros(self.sys_ads_dim)
        for a in da1:
            if a in self.sys_ads2id:
                sys_ads_vec[self.sys_ads2id[a]] = 1
        return sys_ads_vec  # sys_ads_vec dim: 186