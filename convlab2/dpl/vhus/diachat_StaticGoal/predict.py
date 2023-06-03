# -*- coding: utf-8 -*-
from convlab2.policy.vhus.diachat_StaticGoal.vhus_diachat import UserPolicyVHUS
from convlab2.policy.vhus.diachat_StaticGoal.usermanager import UserDataManager
from convlab2.policy.vhus.diachat_StaticGoal.usermodule import VHUS
from convlab2.policy.vhus.diachat_StaticGoal.util import padding_data
from convlab2.util.train_util import to_device
from pprint import pprint
from copy import deepcopy
import numpy as np
import torch
import os
import sys
import json
import random


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    DEVICES = [0, 1]

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)


def sequential(da_seq):
    da = []
    cur_act = None
    for word in da_seq:
        if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '(', ')']:
            continue
        if '-' not in word:
            cur_act = word
        else:
            if cur_act is None:
                continue
            da.append(cur_act + '-' + word)
    return da


def dr(dss):
    return np.array([d for ds in dss for d in ds])


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


if __name__ == '__main__':
    # with open('convlab2/policy/vhus/diachat/config.json', 'r') as f:
    #     config = json.load(f)
    # with open('data/diachat/annotations_goal.json', 'r',
    #           encoding='utf-8') as fp:
    #     full_data = json.load(fp)
    #     manager = UserDataManager()
    #     session = random.choice(full_data)
    #     goals = []
    #     usr_dass = []
    #     sys_dass = []
    #     form_goal = manager.change_goal_form(session.get('goal', []))
    #     print("-"*35 + "Input Goal" + "-"*35)
    #     pprint(form_goal)
    #     logs = session.get('utterances', [])
    #     usr_das, sys_das = [], []
    #     for turn in range(len(logs) // 2):
    #         da_temp = {}
    #         for da in logs[turn * 2].get('annotation'):
    #             act_label = da['act_label']
    #             for dsv in da['slot_values']:
    #                 org(da_temp, dsv, act_label)
    #         usr_das.append(da_temp)
    #         da_temp = {}
    #         for da in logs[turn * 2 + 1].get('annotation'):
    #             act_label = da['act_label']
    #             for dsv in da['slot_values']:
    #                 org(da_temp, dsv, act_label)
    #         sys_das.append(da_temp)
    #     else:
    #         goals.append(form_goal)
    #         usr_dass.append(usr_das)
    #         sys_dass.append(sys_das)
    #     org_goals = [manager.usrgoal2seq(goal) for goal in goals]
    #     org_usr_dass = [[manager.usrda2seq(usr_da, goal) for usr_da in usr_das] for (usr_das, goal)
    #                     in zip(usr_dass, goals)]
    #     org_sys_dass = [[manager.sysda2seq(sys_da, goal) for sys_da in sys_das] for (sys_das, goal)
    #                     in zip(sys_dass, goals)]
    #     goals = [manager.get_goal_id(goal) for goal in org_goals]
    #     usr_dass = [manager.get_usrda_id(usr_das) for usr_das in org_usr_dass]
    #     sys_dass = [manager.get_sysda_id(sys_das) for sys_das in org_sys_dass]
    #     goals_seg, usr_dass_seg, sys_dass_seg = [], [], []

    #     for (goal, usr_das, sys_das) in zip(goals, usr_dass, sys_dass):
    #         goals, usr_dass, sys_dass = [], [], []
    #         for length in range(len(usr_das)):
    #             goals.append(goal)
    #             usr_dass.append([usr_das[idx] for idx in range(length + 1)])
    #             sys_dass.append([sys_das[idx] for idx in range(length + 1)])

    #         goals_seg.append(goals)
    #         usr_dass_seg.append(usr_dass)
    #         sys_dass_seg.append(sys_dass)

    #     idx = range(len(goals_seg))
    #     idx = list(idx)
    #     val_goals, val_usrdas, val_sysdas = dr(np.array(goals_seg)[idx]), dr(np.array(usr_dass_seg)[idx]), dr(
    #         np.array(sys_dass_seg)[idx])
    #     data_valid = (val_goals, val_usrdas, val_sysdas)
    #     data = (data_valid[0], data_valid[1], data_valid[2])

    #     voc_goal_size, voc_usr_size, voc_sys_size = manager.get_voc_size()
    #     user = VHUS(config, voc_goal_size, voc_usr_size, voc_sys_size)
    #     user.load_state_dict(torch.load(
    #         'convlab2/policy/vhus/diachat/save/best_simulator.mdl'))
    #     user.eval()
    #     user = user.cuda()
    #     eos_id = user.usr_decoder.eos_id

    #     batch_input = to_device(padding_data(data))

    #     # select_action
    #     print('-'*35 + 'Select Action' + '-'*35)
    #     usr_a, terminated = user.select_action(batch_input['goals'][0], batch_input['goals_length'][0],
    #                                         batch_input['posts'][0], batch_input['posts_length'][0])
    #     pprint(manager.usrseq2da(manager.id2sentence(usr_a), goals))
    #     print(terminated)

    #     print('-'*35 + 'Predict' + '-'*35)
        user = UserPolicyVHUS(load_from_zip=True)
        user.init_session()
        pprint(user.goal)
        pprint(user.predict())
        pprint(user.terminated)
