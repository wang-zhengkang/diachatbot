# -*- coding: UTF-8 -*-
import os
import json
import numpy as np
from convlab2.policy.vec import Vector
from convlab2.policy.gdpl.diachat.util.lexicalize import delexicalize_da, deflat_da
from convlab2.policy.gdpl.diachat.util.state_structure import belief_state_vectorize
from convlab2.policy.gdpl.diachat.util.domain_act_slot import *

class DiachatVector(Vector):

    def __init__(self, sys_da_file, usr_da_file, character='sys', vocab_size=500):

        self.domains = ['饮食', '行为', '运动', '治疗', '问题', '基本信息']
        self.vocab_size = vocab_size

        # self.da_voc为系统端动作，self.da_voc_opp为用户端动作
        self.da_voc = json.load(open(sys_da_file, encoding='UTF-8'))
        self.da_voc_opp = json.load(open(usr_da_file, encoding='UTF-8'))
        
        self.askfor_ds = json.load(open('convlab2/policy/gdpl/diachat/data/askfor_ds.json', encoding='UTF-8'))
        self.askforsure_ds = json.load(open('convlab2/policy/gdpl/diachat/data/askforsure_ds.json', encoding='UTF-8'))
        
        self.character = character
        self.generate_dict()
        self.cur_domain = None

    def load_composite_actions(self):
        """
        load the composite actions to self.da_voc
        """
        composite_actions_filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'data/multiwoz/da_slot_cnt.json')
        with open(composite_actions_filepath, 'r') as f:
            composite_actions_stats = json.load(f)
            for action in composite_actions_stats:
                if len(action.split(';')) > 1:
                    # append only composite actions as single actions are already in self.da_voc
                    self.da_voc.append(action)

                if len(self.da_voc) == self.vocab_size:
                    break

    def generate_dict(self):
        """
        init the dict for mapping state/action into vector
        """
        # sys_action
        self.act2vec = dict((a, i) for i, a in enumerate(self.da_voc))
        self.vec2act = dict((v, k) for k, v in self.act2vec.items())
        self.da_dim = len(self.da_voc)

        # usr_action
        self.opp2vec = dict((a, i) for i, a in enumerate(self.da_voc_opp))
        self.vec2opp = dict((v, k) for k, v in self.opp2vec.items())
        self.da_opp_dim = len(self.da_voc_opp)

        # belief_state
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
        
        # cur_domain
        self.domain2id = dict((a, i) for i, a in enumerate(self.domains))
        self.id2domain = dict((i, a) for i, a in enumerate(self.domains))
        self.domain_dim = len(self.domains)

        # askfor_dsv 不关注value值
        self.askfor_ds2vec = dict((a, i) for i, a in enumerate(self.askfor_ds))
        self.vec2askfor_ds = dict((v, k) for k, v in self.askfor_ds2vec.items())
        self.askfor_ds_dim = len(self.askfor_ds)

        # askforsure_dsv 不关注value值
        self.askforsure_ds2vec = dict((a, i) for i, a in enumerate(self.askforsure_ds))
        self.vec2askforsure_ds = dict((v, k) for k, v in self.askforsure_ds2vec.items())
        self.askforsure_ds_dim = len(self.askforsure_ds)

        # 
        self.state_dim = self.da_dim + self.da_opp_dim + self.belief_state_dim + \
            + self.domain_dim + self.askfor_ds_dim + self.askforsure_ds_dim + 1
        


    def pointer(self, turn):
        pointer_vector = np.zeros(6 * len(self.db_domains))
        for domain in self.db_domains:
            constraint = turn[domain.lower()]['semi'].items()
            entities = self.db.query(domain.lower(), constraint)
            pointer_vector = self.one_hot_vector(len(entities), domain, pointer_vector)

        return pointer_vector

    def one_hot_vector(self, num, domain, vector):
        """Return number of available entities for particular domain."""
        if domain != 'train':
            idx = self.db_domains.index(domain)
            if num == 0:
                vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
            elif num == 1:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
            elif num == 2:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
            elif num == 3:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
            elif num == 4:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
            elif num >= 5:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
        else:
            idx = self.db_domains.index(domain)
            if num == 0:
                vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
            elif num <= 2:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
            elif num <= 5:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
            elif num <= 10:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
            elif num <= 40:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
            elif num > 40:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

        return vector

    def state_vectorize(self, state):
        """vectorize a state

        Args:
            state (dict):
                Dialog state
            action (tuple):
                Dialog act
        Returns:
            state_vec (np.array):
                Dialog state vector
        """
        self.belief_state = state['belief_state']
        da = state['usr_action']  # da必定为空，这里有点奇怪
        de_act, de_domain, de_act_domain, da = delexicalize_da(da)
        domain_list = []
        if self.character == 'sys':
            action = state['usr_action']
            for intent, domain, slot, value in action:
                if domain not in domain_list:
                    domain_list.append(domain)
        self.cur_domain = domain_list

        usr_act_vec = np.zeros(self.da_opp_dim)
        for a in da:
            if a in self.opp2vec:
                usr_act_vec[self.opp2vec[a]] = 1.

        da = state['sys_action']
        de_act, de_domain, de_act_domain, da = delexicalize_da(da)

        sys_act_vec = np.zeros(self.da_dim)
        for a in da:
            if a in self.act2vec:
                sys_act_vec[self.act2vec[a]] = 1.
        
        cur_domain_vec = np.zeros(self.domain_dim)
        for domain in state["cur_domain"]:
            if domain in self.domains:
                cur_domain_vec[self.domain2id[domain]] = 1.
        
        askfor_dsv_vec = np.zeros(self.askfor_ds_dim)
        for askfor_dsv in state["askfor_dsv"]:
            d = askfor_dsv[0]
            s = askfor_dsv[1]
            ds = '-'.join((d, s))
            if ds in self.askfor_ds2vec:
                askfor_dsv_vec[self.askfor_ds2vec[ds]] = 1.

        askforsure_dsv_vec = np.zeros(self.askforsure_ds_dim)
        for askforsure_dsv in state["askforsure_dsv"]:
            d = askforsure_dsv[0]
            s = askforsure_dsv[1]
            ds = '-'.join((d, s))
            if ds in self.askforsure_ds2vec:
                askforsure_dsv_vec[self.askforsure_ds2vec[ds]] = 1.


        terminated_vec = 1. if state['terminated'] else 0.

        belief_state_vec = belief_state_vectorize(state['belief_state'])
        state_vec = np.r_[
        sys_act_vec,  # sys act+domain+slot
        usr_act_vec,  # usr act+domain+slot
        belief_state_vec,  # belief state
        cur_domain_vec,
        askfor_dsv_vec,
        askforsure_dsv_vec,
        terminated_vec]

        return state_vec

    def dbquery_domain(self, domain):
        """
        query entities of specified domain
        Args:
            domain string:
                domain to query
        Returns:
            entities list:
                list of entities of the specified domain
        """
        constraint = self.state[domain.lower()]['semi'].items()
        return self.db.query(domain.lower(), constraint)

    def action_devectorize(self, action_vec):
        """
        recover an action
        Args:
            action_vec (np.array):
                Dialog act vector
        Returns:
            action (tuple):
                Dialog act
        """
        act_array = []

        
        for i, idx in enumerate(action_vec):
            if idx == 1:
                act_array.append(self.vec2act[i])
        action = deflat_da(act_array)
        return action

    def action_vectorize(self, action):
        _, _, _, da1 = delexicalize_da(action)
        sys_act_vec = np.zeros(self.da_dim)
        for a in da1:
            if a in self.act2vec:
                sys_act_vec[self.act2vec[a]] = 1.  # 把这一部分矩阵化
        return sys_act_vec
