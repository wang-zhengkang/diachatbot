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
skf = KFold(n_splits=10, shuffle=True, random_state=34)
from convlab2.util.train_util import init_logging_handler
from convlab2.task.diachat.goal_generator2 import GoalGenerator
from convlab2.policy.vhus.diachat_DG_BiGRU.usermanager import UserDataManager
from convlab2.policy.vhus.diachat_DG_BiGRU.train import VHUS_Trainer

if __name__ == '__main__':
    with open('convlab2/policy/vhus/diachat_DG_BiGRU/config.json', 'r') as f:
        cfg = json.load(f)
    batchsz = cfg['batchsz']
    epoch = cfg['epoch']
    lr = cfg['lr']
    hu_dim = cfg['hu_dim']
    eu_dim = cfg['eu_dim']
    alpha = cfg['alpha']
    init_logging_handler(cfg['log_dir'])
    manager = UserDataManager()
    goal_gen = GoalGenerator()

    seq_goalss, seq_usr_dass, seq_sys_dass = manager.data_loader_seg()
    data_idx = range(len(seq_goalss))
    total_F1 = 0.
    total_precise = 0.
    total_recall = 0.
    best_F1 = 0.
    best_F1_fold = 1
    worst_F1 = 0.
    worst_F1_fold = 1
    start = int(time.time())
    is_train_all = True
    if is_train_all:
        # 将全部数据集加入训练
        # train_idx为所有数据
        env = VHUS_Trainer(cfg, manager, goal_gen, fold='all', train_idx=[i for i in data_idx], test_idx = [])
        for e in range(cfg['epoch']):
            env.imitating(e, is_train_all)
    else:
        for fold, (train_idx, test_idx) in enumerate(skf.split(data_idx)):
            fold += 1
            logging.info(f"--------------{fold} fold train--------------")
            env = VHUS_Trainer(cfg, manager, goal_gen, fold, train_idx, test_idx)
            best = float('inf')
            for e in range(cfg['epoch']):
                env.imitating(e + 1)
                # best = env.imit_test(e, best)
            env.test()
            if fold == 1:
                best_F1 = env.kfold_info['F1']
                worst_F1 = env.kfold_info['F1']
            else:
                if best_F1 < env.kfold_info['F1']:
                    best_F1 = env.kfold_info['F1']
                    best_F1_fold = fold
                if worst_F1 > env.kfold_info['F1']:
                    worst_F1 = env.kfold_info['F1']
                    worst_F1_fold = fold
            total_F1 += env.kfold_info['F1']
            total_precise += env.kfold_info['precise']
            total_recall += env.kfold_info['recall']
            avg_F1 = total_F1 / fold
            avg_precise = total_precise / fold
            avg_recall = total_recall / fold
            logging.info(f"{fold} fold avg F1:{avg_F1: .6f}")
            logging.info(f"{fold} fold avg precise:{avg_precise: .6f}")
            logging.info(f"{fold} fold avg recall:{avg_recall: .6f}")
        # is_train_all = True
        # env = VHUS_Trainer(cfg, manager, goal_gen, fold='all', train_idx=[i for i in data_idx], test_idx = [])
        # logging.info(f"waiting for train all data...")
        # for e in range(cfg['epoch']):
        #     env.imitating(e, is_train_all)
        #     logging.info(f"{e} epoch complete.")
        # logging.info('<<US>> train all data module saved network to mdl')
        end = int(time.time())
        m, s = divmod(end - start, 60)
        h, m = divmod(m, 60)
        logging.info(f"Train model cost time {h:0>2d}:{m:0>2d}:{s:0>2d}.")
        logging.info(f"--------------Statistics info start---------------")
        logging.info(f'batchsz:{batchsz}')
        logging.info(f'epoch:{epoch}')
        logging.info(f'lr:{lr}')
        logging.info(f'hu_dim:{hu_dim}')
        logging.info(f'eu_dim:{eu_dim}')
        logging.info(f'alpha:{alpha}')
        logging.info(f'random_state:34')
        logging.info('action structure(modify train.test.sequential!):ads* or a-ds')
        logging.info('Bi-GRU:none* or Bi-Decode or Bi-Encode or Bi-De&En')
        logging.info(f"avg F1:{avg_F1: .6f}")
        logging.info(f"best F1:{best_F1: .6f}, best F1 fold:{best_F1_fold}")
        logging.info(f"worst F1:{worst_F1: .6f}, worst F1 fold:{worst_F1_fold}")
        logging.info(f"--------------Statistics info finish--------------")
