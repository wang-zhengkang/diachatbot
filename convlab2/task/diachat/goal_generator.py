'''
rule-based goal generator, need expert knowledge.
'''
import json
import os
import pickle
import random
from collections import Counter
from copy import deepcopy
from pprint import pprint
import numpy as np

from convlab2 import get_root_path

# from convlab2.util.diachat.dbquery import Database

domains = {'基本信息', '问题', '饮食', '运动', '行为', '治疗'}
SORTS = {'current', 'AskForSure', 'AskFor', 'AskHow', 'AskWhy'}


def nomial_sample(counter: Counter):
    return list(counter.keys())[np.argmax(np.random.multinomial(1, list(counter.values())))]


class GoalGenerator:
    """User goal generator."""

    def __init__(self,
                 goal_model_path=os.path.join(get_root_path(), 'data/diachat/goal/new_goal_model.pkl'),
                 corpus_path=None,
                 # sample_info_from_trainset=True,
                 # sample_reqt_from_trainset=False
                 ):
        self.goal_model_path = goal_model_path
        self.corpus_path = corpus_path
        # self.db = Database()
        # self.sample_info_from_trainset = sample_info_from_trainset
        # self.sample_reqt_from_trainset = sample_reqt_from_trainset
        # self.train_database = self.db.query('train', [])
        if os.path.exists(self.goal_model_path):
            self.ind_slot_dist, self.ind_slot_value_dist, self.sorts_combination_dist, self.slots_num_dist, self.slots_combination_dist = pickle.load(
                open(self.goal_model_path, 'rb'))
            print('Loading goal model is done')
        else:
            self._build_goal_model()
            print('Building goal model is done')

    def _build_goal_model(self):
        # 加载数据集
        dialogs = json.load(open(self.corpus_path, encoding='UTF-8'))

        # diachat形式
        # sort: current AskForSure AskFor AskHow AskWhy, 类比于info reqt book
        def _get_dialog_sort(dialog):
            return dialog['domains']

        sorts_combination_list = []
        for d in dialogs:
            sorts_temp = []
            for goal in d['goal']:
                for sort in goal.keys():
                    if sort not in sorts_temp:
                        sorts_temp.append(sort)
            if sorts_temp:
                sorts_combination_list.append(tuple(sorts_temp))
        sorts_combination_cnt = Counter(sorts_combination_list)
        self.sorts_combination_dist = deepcopy(sorts_combination_cnt)
        for sorts in sorts_combination_cnt.keys():
            self.sorts_combination_dist[sorts] = sorts_combination_cnt[sorts] / sum(sorts_combination_cnt.values())

        # independent goal slot distribution
        ind_slot_value_cnt = dict([(sort, {}) for sort in SORTS])
        self.ind_slot_dist = dict([(sort, {}) for sort in SORTS])
        sort_cnt = Counter()
        self.slots_combination_dist = {sort: {} for sort in SORTS}  # 复合slots分布
        self.slots_num_dist = {sort: {} for sort in SORTS}  # slots数量分布

        for d in dialogs:
            slots_dic = {}
            for _, goal in enumerate(d['goal']):
                for sort, sort_args in goal.items():
                    sort_cnt[sort] += 1
                    if sort == 'current':
                        domain = sort_args[0]
                        slot = sort_args[1]
                        value = sort_args[2]
                        if domain not in self.slots_combination_dist[sort]:
                            self.slots_combination_dist[sort][domain] = {}
                            self.slots_num_dist[sort][domain] = {}
                        if sort not in slots_dic:
                            slots_dic[sort] = {}
                        if domain not in slots_dic[sort]:
                            slots_dic[sort][domain] = []
                        if slot not in slots_dic[sort][domain]:
                            slots_dic[sort][domain].append(slot)
                        slots_dic[sort][domain] = sorted(slots_dic[sort][domain])

                        if domain not in ind_slot_value_cnt[sort]:
                            ind_slot_value_cnt[sort][domain] = {}
                        if slot not in ind_slot_value_cnt[sort][domain]:
                            ind_slot_value_cnt[sort][domain][slot] = Counter()
                        ind_slot_value_cnt[sort][domain][slot][value] += 1
                    elif sort == 'AskWhy':
                        why_list = sort_args['why']
                        if not why_list:
                            why_list = [['null', 'null', 'null']]
                        for why in why_list:
                            domain = why[0]
                            slot = why[1]
                            value = why[2]
                            if domain not in self.slots_combination_dist[sort]:
                                self.slots_combination_dist[sort][domain] = {}
                                self.slots_num_dist[sort][domain] = {}
                            if sort not in slots_dic:
                                slots_dic[sort] = {}
                            if domain not in slots_dic[sort]:
                                slots_dic[sort][domain] = []
                            if slot not in slots_dic[sort][domain]:
                                slots_dic[sort][domain].append(slot)
                            slots_dic[sort][domain] = sorted(slots_dic[sort][domain])
                            if domain not in ind_slot_value_cnt[sort]:
                                ind_slot_value_cnt[sort][domain] = {}
                            if slot not in ind_slot_value_cnt[sort][domain]:
                                ind_slot_value_cnt[sort][domain][slot] = Counter()
                            ind_slot_value_cnt[sort][domain][slot][value] += 1
                    else:
                        arg_list = sort_args['args']
                        if not arg_list:
                            arg_list = [['', '', '']]
                        for arg in arg_list:
                            domain = arg[0]
                            slot = arg[1]
                            value = arg[2]
                            if domain not in self.slots_combination_dist[sort]:
                                self.slots_combination_dist[sort][domain] = {}
                                self.slots_num_dist[sort][domain] = {}
                            if sort not in slots_dic:
                                slots_dic[sort] = {}
                            if domain not in slots_dic[sort]:
                                slots_dic[sort][domain] = []
                            if slot not in slots_dic[sort][domain]:
                                slots_dic[sort][domain].append(slot)
                            slots_dic[sort][domain] = sorted(slots_dic[sort][domain])
                            if domain not in ind_slot_value_cnt[sort]:
                                ind_slot_value_cnt[sort][domain] = {}
                            if slot not in ind_slot_value_cnt[sort][domain]:
                                ind_slot_value_cnt[sort][domain][slot] = Counter()
                            ind_slot_value_cnt[sort][domain][slot][value] += 1

            for sort, ds_dic in slots_dic.items():
                for domain, slots in ds_dic.items():
                    self.slots_combination_dist[sort][domain].setdefault(tuple(slots), 0)
                    self.slots_combination_dist[sort][domain][tuple(slots)] += 1
                    # 记录slots数量的分布
                    self.slots_num_dist[sort][domain].setdefault(len(slots), 0)
                    self.slots_num_dist[sort][domain][len(slots)] += 1

        self.ind_slot_value_dist = deepcopy(ind_slot_value_cnt)
        for sort in SORTS:
            for domain, slots in ind_slot_value_cnt[sort].items():
                for slot in slots:
                    if domain not in self.ind_slot_dist[sort]:
                        self.ind_slot_dist[sort][domain] = {}
                    if slot not in self.ind_slot_dist[sort][domain]:
                        self.ind_slot_dist[sort][domain][slot] = {}
                    self.ind_slot_dist[sort][domain][slot] = sum(ind_slot_value_cnt[sort][domain][slot].values()) / \
                                                             sort_cnt[sort]
                    slot_total = sum(ind_slot_value_cnt[sort][domain][slot].values())
                    for val in self.ind_slot_value_dist[sort][domain][slot]:
                        # value所在slot的占比
                        self.ind_slot_value_dist[sort][domain][slot][val] = ind_slot_value_cnt[sort][domain][slot][
                                                                                val] / slot_total

        pickle.dump((self.ind_slot_dist, self.ind_slot_value_dist, self.sorts_combination_dist, self.slots_num_dist,
                     self.slots_combination_dist),
                    open(self.goal_model_path, 'wb'))

    def _get_sort_goal(self, sort):
        cnt_slot = self.ind_slot_dist[sort]
        cnt_slot_value = self.ind_slot_value_dist[sort]
        # while True:
        #     domain_goal = {sort: {}}
        #     slots = random.choices(list(self.slots_combination_dist[sort]['info'].keys()),
        #                            list(self.slots_combination_dist[sort]['info'].values()))[0]
        #     for slot in slots:
        #         domain_goal[sort][slot] = nomial_sample(cnt_slot_value['info'][slot])

        

    def get_user_goal(self):
        sorts_combination = ()
        while len(sorts_combination) <= 0:
            sorts_combination = nomial_sample(self.sorts_combination_dist)
        user_goal = {sort: self._get_sort_goal(sort) for sort in sorts_combination}
        assert len(user_goal.keys()) > 0


if __name__ == '__main__':
    goal_generator = GoalGenerator(corpus_path=os.path.join(get_root_path(), 'data/diachat/annotations_goal.json'))
    pprint(goal_generator.get_user_goal())
