# -*- coding: utf-8 -*-
"""
@author: truthless
"""
import os
import json
import logging
import sys
import time
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

from sklearn.model_selection import KFold
skf = KFold(n_splits=10,shuffle=True,random_state=2023)
from convlab2.util.train_util import init_logging_handler
from convlab2.task.diachat.goal_generator2 import GoalGenerator
from convlab2.policy.vhus.diachat_DynamicGoal.usermanager import UserDataManager
from convlab2.policy.vhus.diachat_DynamicGoal.train import VHUS_Trainer

if __name__ == '__main__':
    with open('convlab2/policy/vhus/diachat_DynamicGoal/config.json', 'r') as f:
        cfg = json.load(f)
    is_train_all = True
    init_logging_handler(cfg['log_dir'])
    manager = UserDataManager()
    goal_gen = GoalGenerator()

    seq_goalss, seq_usr_dass, seq_sys_dass = manager.data_loader_seg()
    data_idx = range(len(seq_goalss))
    total_F1 = 0
    total_precise = 0
    total_recall = 0
    start = int(time.time())
    if is_train_all:
        # 将全部数据集加入训练
        # train_idx为所有数据
        env = VHUS_Trainer(cfg, manager, goal_gen, fold='all', train_idx=[i for i in data_idx], test_idx = [])
        for e in range(cfg['epoch']):
            env.imitating(e, is_train_all)
    else:
        for fold, (train_idx, test_idx) in enumerate(skf.split(data_idx)):
            fold += 1
            logging.info(f"--------------第{fold}折交叉训练--------------")
            env = VHUS_Trainer(cfg, manager, goal_gen, fold, train_idx, test_idx)
            # best = float('inf')
            for e in range(cfg['epoch']):
                env.imitating(e + 1)
                # best = env.imit_test(e, best)
            env.test()
            total_F1 += env.kfold_info['F1']
            total_precise += env.kfold_info['precise']
            total_recall += env.kfold_info['recall']
            avg_F1 = total_F1 / fold
            avg_precise = total_precise / fold
            avg_recall = total_recall / fold
            logging.info(f"{fold}折平均F1:{avg_F1: .6f}")
            logging.info(f"{fold}折平均precise:{avg_precise: .6f}")
            logging.info(f"{fold}折平均recall:{avg_recall: .6f}")
        is_train_all = True
        env = VHUS_Trainer(cfg, manager, goal_gen, fold='all', train_idx=[i for i in data_idx], test_idx = [])
        for e in range(cfg['epoch']):
            env.imitating(e, is_train_all)

    end = int(time.time())
    m, s = divmod(end - start, 60)
    print(f"Train model cost {m}min {s}s.")
    