# -*- coding: UTF-8 -*-
import collections
import json
import os
import pickle
import random
import numpy as np
from pprint import pprint

SORTS = ['current', 'AskForSure', 'AskFor', 'AskHow', 'AskWhy', 'Chitchat']
DOMAINS = ['问题', '饮食', '行为', '运动', '治疗', '基本信息']


class UserDataManager(object):

    def __init__(self):
        self.__org_goals = None
        self.__org_usr_dass = None
        self.__org_sys_dass = None

        self.__goalss = None
        self.__usr_dass = None
        self.__sys_dass = None

        self.__goalss_seg = None
        self.__usr_dass_seg = None
        self.__sys_dass_seg = None

        self.__voc_goal = None
        self.__voc_usr = None
        self.__voc_usr_rev = None
        self.__voc_sys = None

        self.pad = '<PAD>'
        self.unk = '<UNK>'
        self.sos = '<SOS>'
        self.eos = '<EOS>'
        self.special_words = [self.pad, self.unk, self.sos, self.eos]

        self.voc_goal, self.voc_usr, self.voc_sys = self.vocab_loader()

    @staticmethod
    def usrgoal2seq(goal: dict):
        ret = []
        for sort in SORTS:
            if goal[sort]:
                ret.append(sort)
                ret.append('(')
                for dsdonev in goal[sort]:
                    ret.append(dsdonev[0])
                ret.append(')')
        return ret

    def get_voc_size(self):
        return len(self.voc_goal), len(self.voc_usr), len(self.voc_sys)

    def vocab_loader(self):
        if self.__voc_goal is None or self.__voc_usr is None or self.__voc_usr_rev is None or self.__voc_sys is None:
            vocab_path = 'data/diachat/goal/vocab_DG_BiGRU.pkl'
            if os.path.exists(vocab_path):
                with open(vocab_path, 'rb') as f:
                    self.__voc_goal, self.__voc_usr, self.__voc_sys = pickle.load(f)
                print('load voc ok')
            else:
                goalss, usr_dass, sys_dass = self.org_data_loader()
                counter = collections.Counter()
                for goals in goalss:
                    for goal in goals:
                        for word in goal:
                            counter[word] += 1
                word_list = self.special_words + [x for x in counter.keys()]
                self.__voc_goal = {x: i for i, x in enumerate(word_list)}

                counter = collections.Counter()
                for usr_das in usr_dass:
                    for usr_da in usr_das:
                        for word in usr_da:
                            counter[word] += 1
                word_list = self.special_words + [x for x in counter.keys()]
                self.__voc_usr = {x: i for i, x in enumerate(word_list)}

                counter = collections.Counter()
                for sys_das in sys_dass:
                    for sys_da in sys_das:
                        for word in sys_da:
                            counter[word] += 1
                word_list = self.special_words + [x for x in counter.keys()]
                self.__voc_sys = {x: i for i, x in enumerate(word_list)}
                with open(vocab_path, 'wb') as f:
                    pickle.dump((self.__voc_goal, self.__voc_usr, self.__voc_sys), f)
                print('voc build ok')
            # 缩进分割线
            self.__voc_usr_rev = {val: key for (key, val) in self.__voc_usr.items()}
            self.__voc_sys_rev = {val: key for (key, val) in self.__voc_sys.items()}
            return self.__voc_goal, self.__voc_usr, self.__voc_sys

    @staticmethod
    # multiwoz中此方法的作用是将所有单词小写，该方法替换为change_da_form
    def ref_data2stand(da):
        return da

    @staticmethod
    def change_da_form(da):
        return da

    @staticmethod
    def change_goal_form(goals):
        user_goal = dict([(sort, []) for sort in SORTS])
        for goal in goals:
            for sort, sort_args in goal.items():
                if sort == 'current' or sort == 'Chitchat':
                    domain = sort_args[0]
                    slot = sort_args[1]
                    value = sort_args[2]
                    done = str(sort_args[3])
                    temp = [domain + '-' + slot + '-' + done, value]
                    user_goal[sort].append(temp)
                elif sort == 'AskWhy':
                    why_list = sort_args['why']
                    done = str(sort_args['done'])
                    if not why_list:
                        user_goal[sort].append(['none-none' + '-' + done, 'none'])
                        break
                    for why in why_list:
                        domain = why[0]
                        slot = why[1]
                        value = why[2]
                        temp = [domain + '-' + slot + '-' + done, value]
                        user_goal[sort].append(temp)
                # elif sort == 'Chitchat':
                #     done = str(sort_args['done'])
                #     user_goal[sort].append('none-none', 'none')
                else:
                    done = str(sort_args['done'])
                    arg_list = sort_args['args']
                    if arg_list == [[]] or arg_list == [["", "", ""]]:
                        user_goal[sort].append(['none-none' + '-' + done, 'none'])
                        break
                    for arg in arg_list:
                        domain = arg[0]
                        slot = arg[1]
                        value = arg[2]
                        temp = [domain + '-' + slot + '-' + done, value]
                        user_goal[sort].append(temp)
        return user_goal

    @staticmethod
    def usrda2seq(usr_da: dict):
        ret = []
        for act, dsv_list in usr_da.items():
            for dsv in dsv_list:
                ret.append(act + '-' + dsv[0])
        return ret
        # ret = []
        # for act, dsv_list in usr_da.items():
        #     ret.append(act)
        #     ret.append('(')
        #     for dsv in dsv_list:
        #         ds = dsv[0]
        #         ret.append(ds)
        #     ret.append(')')
        # return ret

    @staticmethod
    def sysda2seq(sys_da: dict):
        ret = []
        for act, dsv_list in sys_da.items():
            for dsv in dsv_list:
                ret.append(act + '-' + dsv[0])
        return ret
        # ret = []
        # for act, dsv_list in sys_da.items():
        #     ret.append(act)
        #     ret.append('(')
        #     for dsv in dsv_list:
        #         ret.append(dsv[0])
        #     ret.append(')')
        # return ret
    
    @staticmethod
    def usrseq2da(usr_seq: list, goal: dict):
        def sequential(da_seq):
            # a-ds
            # da = []
            # cur_act = None
            # for word in da_seq:
            #     if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '(', ')']:
            #         continue
            #     if '-' not in word:
            #         cur_act = word
            #     else:
            #         if cur_act is None:
            #             continue
            #         da.append(cur_act + '-' + word)
            # return da

            # ads
            da = []
            cur_act = None
            for word in da_seq:
                if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
                    continue
                da.append(word)
            return da
        ret = sequential(usr_seq)
        return ret

    @staticmethod
    def da_list_form_to_dict_form(da):
        '''
        convert the da in list form to da in dict form
        for example, [['Inform', 'Train', 'Leave', '09:00']] -> {'train-inform': [['leave', '09:00']]}
        '''
        ret = {}
        for [sort, domain, slot, value] in da:
            ds = domain + '-' + slot
            if sort in ret:
                ret[sort].append([ds, value])
            else:
                ret[sort] = [[ds, value]]
        return ret
    
    @staticmethod
    def da_dict_form_to_list_form(da):
        '''
        convert da in dict form to da in list form
        for example, {'restaurant-inform': [['area', 'west'], ['price', 'cheap']]} -> 
                    [['Inform', 'Restaurant', 'Area', 'west'], ['Inform', 'Restaurant', 'Price', 'cheap']]
        '''
        # no_arg: Accept, AskWhy, Assure, Deny, Chitchat
        ret = []
        for act, dss in da.items():
            if act in ['Accept', 'AskWhy', 'Assure', 'Deny', 'Chitchat']:
                ads = []
                ads.append(act)
                ads.append('none')
                ads.append('none')
                ret.append(ads)
                continue
            for ds in dss:
                ads = []
                domain, slot = ds.split('-')
                ads.append(act)
                ads.append(domain)
                ads.append(slot)
                ret.append(ads)

        
        return ret

    def org_data_loader(self):
        def org(da_temp: dict, dsv: dict, act_label):
            domain = dsv['domain'] if dsv['domain'] else 'none'
            slot = dsv['slot'] if dsv['slot'] else 'none'
            value = dsv['value'] if dsv['value'] else 'none'
            key = act_label
            ds = domain + '-' + slot
            dsv_list = [ds, value]
            if key not in da_temp:
                da_temp[key] = [dsv_list]
            else:
                da_temp[key].append(dsv_list)

        if self.__org_goals is None or self.__org_usr_dass is None or self.__org_sys_dass is None:
            with open('data/diachat/annotations_goal.json', 'r',
                      encoding='utf-8') as fp:
                full_data = json.load(fp)
            goalss = []
            usr_dass = []
            sys_dass = []
            for session in full_data:
                usr_das, sys_das ,goals = [], [], []
                goal = self.change_goal_form(session.get('goal', []))
                goals.append(goal)
                logs = session.get('utterances', [])
                for turn in range(len(logs) // 2):
                    da_temp = {}
                    # multiWOZ里da代表domain act
                    for da in logs[turn * 2].get('annotation'):
                        act_label = da['act_label']
                        for dsv in da['slot_values']:
                            org(da_temp, dsv, act_label)
                    usr_das.append(da_temp)

                    usrstate_temp = {}
                    # multiWOZ里da代表domain act
                    if len(goals) < len(logs) // 2:
                        usrstate_temp = self.change_goal_form(logs[turn * 2].get('user_state', []))
                        goals.append(usrstate_temp)
                    
                    da_temp = {}
                    for da in logs[turn * 2 + 1].get('annotation'):
                        act_label = da['act_label']
                        for dsv in da['slot_values']:
                            org(da_temp, dsv, act_label)
                    sys_das.append(da_temp)
                goalss.append(goals)
                usr_dass.append(usr_das)
                sys_dass.append(sys_das)
            self.__org_goalss = [[UserDataManager.usrgoal2seq(goal) for goal in goals] for goals in goalss]
            self.__org_usr_dass = [[UserDataManager.usrda2seq(usr_da) for usr_da in usr_das] for usr_das in usr_dass]
            self.__org_sys_dass = [[UserDataManager.sysda2seq(sys_da) for sys_da in sys_das] for sys_das in sys_dass]
        return self.__org_goalss, self.__org_usr_dass, self.__org_sys_dass

    def data_loader_seg(self):
        if self.__goalss_seg is None or self.__usr_dass_seg is None or self.__sys_dass_seg is None:
            self.data_loader()
            self.__goalss_seg, self.__usr_dass_seg, self.__sys_dass_seg = [], [], []

            for (goals, usr_das, sys_das) in zip(self.__goalss, self.__usr_dass, self.__sys_dass):
                goalss, usr_dass, sys_dass = [], [], []
                for length in range(len(usr_das)):
                    goalss.append([goals[idx] for idx in range(length + 1)])
                    usr_dass.append([usr_das[idx] for idx in range(length + 1)])
                    sys_dass.append([sys_das[idx] for idx in range(length + 1)])

                self.__goalss_seg.append(goalss)
                self.__usr_dass_seg.append(usr_dass)
                self.__sys_dass_seg.append(sys_dass)

        assert len(self.__goalss_seg) == len(self.__usr_dass_seg)
        assert len(self.__goalss_seg) == len(self.__sys_dass_seg)
        return self.__goalss_seg, self.__usr_dass_seg, self.__sys_dass_seg

    def data_loader(self):
        if self.__goalss is None or self.__usr_dass is None or self.__sys_dass is None:
            org_goalss, org_usr_dass, org_sys_dass = self.org_data_loader()
            self.__goalss = [self.get_goal_id(goals) for goals in org_goalss]
            self.__usr_dass = [self.get_usrda_id(usr_das) for usr_das in org_usr_dass]
            self.__sys_dass = [self.get_sysda_id(sys_das) for sys_das in org_sys_dass]
        assert len(self.__goalss) == len(self.__usr_dass)
        assert len(self.__goalss) == len(self.__sys_dass)
        return self.__goalss, self.__usr_dass, self.__sys_dass

    def get_goal_id(self, goals):
        if type(goals[0]) == list:
            return [[self.voc_goal.get(word, self.voc_goal[self.unk]) for word in goal] for goal in goals]
        else:
            return [self.voc_goal.get(word, self.voc_goal[self.unk]) for word in goals]

    def get_sysda_id(self, sys_das):
        return [[self.voc_sys.get(word, self.voc_sys[self.unk]) for word in sys_da] for sys_da in sys_das]

    def get_usrda_id(self, usr_das):
        return [[self.voc_usr[self.sos]] + [self.voc_usr.get(word, self.voc_usr[self.unk]) for word in usr_da] + [
            self.voc_usr[self.eos]]
                for usr_da in usr_das]

    @staticmethod
    def train_test_val_split_seg(goals_seg, usr_dass_seg, sys_dass_seg, test_size=0.1, val_size=0.1):
        def dr(dss):
            return np.array([d for ds in dss for d in ds])

        idx = range(len(goals_seg))
        random.seed(2023)
        idx_test = random.sample(idx, int(len(goals_seg) * test_size),)
        idx_train = list(set(idx) - set(idx_test))
        idx_val = random.sample(idx_train, int(len(goals_seg) * val_size))
        idx_train = list(set(idx_train) - set(idx_val))
        idx_train = random.sample(idx_train, len(idx_train))
        return dr(np.array(goals_seg)[idx_train]), dr(np.array(usr_dass_seg)[idx_train]), dr(
            np.array(sys_dass_seg)[idx_train]), \
            dr(np.array(goals_seg)[idx_test]), dr(np.array(usr_dass_seg)[idx_test]), dr(
            np.array(sys_dass_seg)[idx_test]), \
            dr(np.array(goals_seg)[idx_val]), dr(np.array(usr_dass_seg)[idx_val]), dr(np.array(sys_dass_seg)[idx_val])

    def id2sentence(self, ids):
        sentence = [self.__voc_usr_rev[id] for id in ids]
        return sentence
    def sysid2sentence(self, ids):
        sentence = [self.__voc_sys_rev[id] for id in ids]
        return sentence
    
    @staticmethod
    def kfold_date_process(goalss_seg, usr_dass_seg, sys_dass_seg, train_idx, test_idx):
        def dr(dss):
            return np.array([d for ds in dss for d in ds])
        idx_train = train_idx
        idx_test = test_idx
        return dr(np.array(goalss_seg)[idx_train]), dr(np.array(usr_dass_seg)[idx_train]), dr(
            np.array(sys_dass_seg)[idx_train]), dr(np.array(goalss_seg)[idx_test]), dr(np.array(usr_dass_seg)[idx_test]), dr(np.array(sys_dass_seg)[idx_test])