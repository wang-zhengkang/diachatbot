import os
import json
import numpy as np
from convlab2.dpl.vec import Vector
from convlab2.dpl.etc.util.lexicalize import delexicalize_da, deflat_da
from convlab2.dpl.etc.util.state_structure import belief_state_vectorize
from convlab2.dpl.etc.util.domain_act_slot import *


class DiachatVector(Vector):

    def __init__(self, vocab_size=500):
        sys_da_file = 'convlab2/dpl/etc/data/sys_da.json'
        usr_da_file = 'convlab2/dpl/etc/data/usr_da.json'
        inform_ds_file = 'convlab2/dpl/etc/data/inform_ds.json'
        askhow_ds_file = 'convlab2/dpl/etc/data/askhow_ds.json'
        askwhy_ds_file = 'convlab2/dpl/etc/data/askwhy_ds.json'
        askfor_ds_file = 'convlab2/dpl/etc/data/askfor_ds.json'
        askforsure_ds_file = 'convlab2/dpl/etc/data/askforsure_ds.json'

        self.sys_da_voc = json.load(open(sys_da_file, encoding='UTF-8'))
        self.usr_da_voc = json.load(open(usr_da_file, encoding='UTF-8'))
        self.inform_ds = json.load(open(inform_ds_file, encoding='UTF-8'))
        self.askhow_ds = json.load(open(askhow_ds_file, encoding='UTF-8'))
        self.askwhy_ds = json.load(open(askwhy_ds_file, encoding='UTF-8'))
        self.askfor_ds = json.load(open(askfor_ds_file, encoding='UTF-8'))
        self.askforsure_ds = json.load(open(askforsure_ds_file, encoding='UTF-8'))

        self.domains = domains
        self.vocab_size = vocab_size

        self.generate_dict()

    def generate_dict(self):
        """
        init the dict for mapping state/action into vector
        """

        # sys_action
        self.sys_da2vec = dict((a, i) for i, a in enumerate(self.sys_da_voc))
        self.vec2sys_da = dict((v, k) for k, v in self.sys_da2vec.items())
        self.sys_da_dim = len(self.sys_da_voc)

        # usr_action
        self.usr_da2vec = dict((a, i) for i, a in enumerate(self.usr_da_voc))
        self.vec2usr_da = dict((v, k) for k, v in self.usr_da2vec.items())
        self.usr_da_dim = len(self.usr_da_voc)

        # cur_domain
        self.domain2id = dict((a, i) for i, a in enumerate(self.domains))
        self.id2domain = dict((i, a) for i, a in enumerate(self.domains))
        self.cur_domain_dim = len(self.domains)

        # inform_ds
        self.inform_ds2vec = dict((a, i) for i, a in enumerate(self.inform_ds))
        self.vec2inform_ds = dict((v, k) for k, v in self.inform_ds2vec.items())
        self.inform_ds_dim = len(self.inform_ds)

        # askhow_ds
        self.askhow_ds2vec = dict((a, i) for i, a in enumerate(self.askhow_ds))
        self.vec2askhow_ds = dict((v, k) for k, v in self.askhow_ds2vec.items())
        self.askhow_ds_dim = len(self.askhow_ds)

        # askwhy_ds
        self.askwhy_ds2vec = dict((a, i) for i, a in enumerate(self.askwhy_ds))
        self.vec2askwhy_ds = dict((v, k) for k, v in self.askwhy_ds2vec.items())
        self.askwhy_ds_dim = len(self.askwhy_ds)

        # askfor_ds
        self.askfor_ds2vec = dict((a, i) for i, a in enumerate(self.askfor_ds))
        self.vec2askfor_ds = dict((v, k) for k, v in self.askfor_ds2vec.items())
        self.askfor_ds_dim = len(self.askfor_ds)

        # askforsure_ds
        self.askforsure_ds2vec = dict((a, i) for i, a in enumerate(self.askforsure_ds))
        self.vec2askforsure_ds = dict((v, k) for k, v in self.askforsure_ds2vec.items())
        self.askforsure_ds_dim = len(self.askforsure_ds)

        # belief_state
        self.belief_state_domainslot2id = dict()  # 作用不明
        self.belief_state_id2domainslot = dict()  # 作用不明
        self.belief_state_dim = 0
        for domain in domain_slot.keys():
            if domain == '基本信息':
                self.belief_state_domainslot2id[domain] = domain_slot2id[domain]
                self.belief_state_id2domainslot[domain] = id2domain_slot[domain]
                self.belief_state_dim += len(domain_slot2id[domain])
            else:
                self.belief_state_domainslot2id[domain] = {'现状': domain_slot2id[domain],
                                                           '解释': domain_slot2id[domain]}
                self.belief_state_id2domainslot[domain] = {'现状': id2domain_slot[domain],
                                                           '解释': id2domain_slot[domain]}
                self.belief_state_dim += 2 * len(domain_slot2id[domain])
                if domain != '问题':
                    self.belief_state_domainslot2id[domain]['建议'] = domain_slot2id[domain]
                    self.belief_state_id2domainslot[domain]['建议'] = id2domain_slot[domain]
                    self.belief_state_dim += len(domain_slot2id[domain])

        # terminate 维度+1
        self.state_dim = self.sys_da_dim + self.usr_da_dim + self.cur_domain_dim + \
            self.inform_ds_dim  + self.askhow_ds_dim + self.askwhy_ds_dim + \
            self.askfor_ds_dim + self.askforsure_ds_dim + self.belief_state_dim + 1


    def state_vectorize(self, state):
        """
        vectorize a state

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
        da = state['usr_da']
        de_act, de_domain, de_act_domain, da = delexicalize_da(da)
        domain_list = []

        usr_da_vec = np.zeros(self.usr_da_dim)
        for a in da:
            if a in self.usr_da2vec:
                usr_da_vec[self.usr_da2vec[a]] = 1.

        da = state['sys_da']
        de_act, de_domain, de_act_domain, da = delexicalize_da(da)

        sys_da_vec = np.zeros(self.sys_da_dim)
        for a in da:
            if a in self.sys_da2vec:
                sys_da_vec[self.sys_da2vec[a]] = 1.

        cur_domain_vec = np.zeros(self.cur_domain_dim)
        for domain in state["cur_domain"]:
            if domain in self.domains:
                cur_domain_vec[self.domain2id[domain]] = 1.

        inform_ds_vec = np.zeros(self.inform_ds_dim)
        for inform_dsv in state["inform_ds"]:
            d = inform_dsv[0]
            s = inform_dsv[1]
            ds = '-'.join((d, s))
            if ds in self.inform_ds2vec:
                inform_ds_vec[self.inform_ds2vec[ds]] = 1.
        
        askhow_ds_vec = np.zeros(self.askhow_ds_dim)
        for askhow_dsv in state["askhow_ds"]:
            d = askhow_dsv[0]
            s = askhow_dsv[1]
            ds = '-'.join((d, s))
            if ds in self.askhow_ds2vec:
                askhow_ds_vec[self.askhow_ds2vec[ds]] = 1.
        
        askwhy_ds_vec = np.zeros(self.askwhy_ds_dim)
        for askwhy_dsv in state["askwhy_ds"]:
            d = askwhy_dsv[0]
            s = askwhy_dsv[1]
            ds = '-'.join((d, s))
            if ds in self.askwhy_ds2vec:
                askwhy_ds_vec[self.askwhy_ds2vec[ds]] = 1.

        askfor_ds_vec = np.zeros(self.askfor_ds_dim)
        for askfor_dsv in state["askfor_ds"]:
            d = askfor_dsv[0]
            s = askfor_dsv[1]
            ds = '-'.join((d, s))
            if ds in self.askfor_ds2vec:
                askfor_ds_vec[self.askfor_ds2vec[ds]] = 1.

        askforsure_ds_vec = np.zeros(self.askforsure_ds_dim)
        for askforsure_dsv in state["askforsure_ds"]:
            d = askforsure_dsv[0]
            s = askforsure_dsv[1]
            ds = '-'.join((d, s))
            if ds in self.askforsure_ds2vec:
                askforsure_ds_vec[self.askforsure_ds2vec[ds]] = 1.

        terminate_vec = 1. if state['terminate'] else 0.

        belief_state_vec = belief_state_vectorize(state['belief_state'])
        state_vec = np.r_[
            sys_da_vec,
            usr_da_vec,
            belief_state_vec,
            cur_domain_vec,
            inform_ds_vec,
            askhow_ds_vec,
            askwhy_ds_vec,
            askfor_ds_vec,
            askforsure_ds_vec,
            terminate_vec
        ]

        return state_vec


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
                act_array.append(self.vec2sys_da[i])
        action = deflat_da(act_array)
        return action

    def action_vectorize(self, action):
        _, _, _, da = delexicalize_da(action)
        sys_da_vec = np.zeros(self.sys_da_dim)
        for a in da:
            if a in self.sys_da2vec:
                sys_da_vec[self.sys_da2vec[a]] = 1.
        return sys_da_vec
