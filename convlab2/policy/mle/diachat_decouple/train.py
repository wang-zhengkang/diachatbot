import os, sys
root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

import torch
import logging
import torch.nn as nn
import json
import time
from convlab2.policy.rlmodule import MultiDiscretePolicy
from convlab2.policy.mle.diachat_decouple.vector_diachat import DiachatVector
from convlab2.policy.mle.diachat_decouple.loader import PolicyDataLoaderDiachat
from convlab2.util.train_util import to_device, init_logging_handler

from sklearn.model_selection import KFold
skf = KFold(n_splits=10,shuffle=True,random_state=2023)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLE_Trainer():

    def __init__(self, cfg, fold, train_idx, test_idx, is_train_all=False):
        sys_ads_file = 'convlab2/policy/mle/diachat_decouple/processed_data/sys_ads.json'
        usr_ads_file = 'convlab2/policy/mle/diachat_decouple/processed_data/usr_ads.json'
        sys_ad_file = 'convlab2/policy/mle/diachat_decouple/processed_data/sys_ad.json'
        usr_ad_file = 'convlab2/policy/mle/diachat_decouple/processed_data/usr_ad.json'
        vector = DiachatVector(sys_ads_file, usr_ads_file, sys_ad_file, usr_ad_file)
        
        self.save_dir = cfg['save_dir']
        self.print_per_batch = cfg['print_per_batch']
        self.save_per_epoch = cfg['save_per_epoch']
        self.batchsz = cfg['batchsz']
        self.epoch = cfg['epoch']
        self.lr = cfg['lr']
        self.h_dim = cfg['h_dim']
        logging.info(f'batchsz:{self.batchsz}')
        logging.info(f'epoch:{self.epoch}')
        logging.info(f'lr:{self.lr}')
        logging.info(f'h_dim:{self.h_dim}')
        self.fold = fold
        self.kfold_info = {
            'F1': float,
            'precise': float,
            'recall': float
        }

        self.manager = PolicyDataLoaderDiachat(vector)
        self.data_train = self.manager.create_dataset('train', cfg['batchsz'], train_idx)
        if not is_train_all:
            self.data_test = self.manager.create_dataset('test', cfg['batchsz'], test_idx)
        

        self.policy = MultiDiscretePolicy(vector.state_dim, cfg['h_dim'], vector.sys_ads_dim).to(
            device=DEVICE)  # 构建一个神经网络，入度为551，出度为186
        self.policy.eval()
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg['lr'])  # torch.optim指定以adam作为优化器
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()  # 指定损失函数为多标签分类损失

    def policy_loop(self, data):
        s, target_a = to_device(data)
        a_weights = self.policy(s)  # 进行一次训练，得到a_weights

        loss_a = self.multi_entropy_loss(a_weights, target_a)  # 计算损失
        return loss_a

    def imitating(self, epoch, is_train_all=False):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)模仿学习（行为模仿）
        """
        self.policy.train()
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            loss_a = self.policy_loop(data)  # 进行一次训练并返回损失
            a_loss += loss_a.item()
            loss_a.backward()  # 通过反向传播过程来实现可训练参数的更新
            self.policy_optim.step()

            if (i + 1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch  # 400轮训练后平均损失
                logging.debug('<<DPL-MLE>> epoch {}, iter {}, loss_a:{}'.format(epoch, i, a_loss))
                a_loss = 0.

        if epoch == self.epoch:
            self.save(self.save_dir, is_train_all)  # 保存一下训练的模型，以mdl模式
        self.policy.eval()

    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the policy fit on the training dataset
        """
        a_loss = 0.
        for i, data in enumerate(self.data_valid):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()

        a_loss /= len(self.data_valid)
        logging.debug('<<DPL-MLE>> val, epoch {}, loss_a:{}'.format(epoch, a_loss))
        if a_loss < best:
            logging.info('<<DPL-MLE>> best model saved')
            best = a_loss
            self.save(self.save_dir, 'best')

        a_loss = 0.
        for i, data in enumerate(self.data_test):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()

        a_loss /= len(self.data_test)
        logging.debug('<<DPL-MLE>> test, epoch {}, loss_a:{}'.format(epoch, a_loss))
        return best

    def test(self):
        def f1(a, target):
            TP, FP, FN = 0, 0, 0
            real = target.nonzero().tolist()
            predict = a.nonzero().tolist()
            for item in real:
                if item in predict:
                    TP += 1
                else:
                    FN += 1
            for item in predict:
                if item not in real:
                    FP += 1
            return TP, FP, FN

        a_TP, a_FP, a_FN = 0, 0, 0
        for i, data in enumerate(self.data_test):
            s, target_a = to_device(data)
            a_weights = self.policy(s)
            a = a_weights.ge(0)  # 关系运算符，大于等于的意思
            # TODO: fix batch F1
            TP, FP, FN = f1(a, target_a)
            a_TP += TP
            a_FP += FP
            a_FN += FN

        precise = a_TP / (a_TP + a_FP)
        recall = a_TP / (a_TP + a_FN)
        F1 = 2 * precise * recall / (precise + recall + 0.0001)
        self.kfold_info['F1'] = F1
        self.kfold_info['precise'] = precise
        self.kfold_info['recall'] = recall
        logging.info(f'a_TP={a_TP}, a_FP={a_FP}, a_FN={a_FN}, F1={F1: .6f}')
        logging.info(f'precise={precise: .6f}, recall={recall: .6f}')

    def save(self, directory, is_train_all=False):
        if is_train_all:
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save(self.policy.state_dict(), directory + '/' + 'train_all_data' + '_mle.pol.mdl')

            logging.info('<<DPL-MLE>> saved {} fold network to mdl'.format(self.fold))
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save(self.policy.state_dict(), directory + '/' + str(self.fold) + '_fold_mle.pol.mdl')

            logging.info('<<DPL-MLE>> saved {} fold network to mdl'.format(self.fold))

    def load(self, filename='save/best'):
        policy_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_mle.pol.mdl')
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))


if __name__ == '__main__':
    with open('convlab2/policy/mle/diachat_decouple/config.json', 'r') as f:
        cfg = json.load(f)
    with open('data/diachat/annotations_goal.json', 'r') as fp:
        data = json.load(fp)
        data_idx = range(len(data))

    # True: 直接将全部数据集加入训练   False: 进行十折交叉验证后再将全部数据集加入训练
    is_train_all = False
    init_logging_handler(cfg['log_dir'])
    total_F1 = 0.
    total_precise = 0.
    total_recall = 0.
    start = int(time.time())
    if is_train_all:
        # 将全部数据集加入训练  train_idx为所有数据
        agent = MLE_Trainer(cfg, fold='all', train_idx=[i for i in data_idx], test_idx = [], is_train_all=True)
        for e in range(cfg['epoch']):
            agent.imitating(e+1, is_train_all)
    else:
        for fold, (train_idx, test_idx) in enumerate(skf.split(data_idx)):
            fold += 1
            agent = MLE_Trainer(cfg, fold, train_idx, test_idx)
            logging.info(f"--------------第{fold}折交叉训练--------------")
            for e in range(cfg['epoch']):
                agent.imitating(e+1)
            agent.test()
            total_F1 += agent.kfold_info['F1']
            total_precise += agent.kfold_info['precise']
            total_recall += agent.kfold_info['recall']
            avg_F1 = total_F1 / fold
            avg_precise = total_precise / fold
            avg_recall = total_recall / fold
            logging.info(f"{fold}折平均F1:{avg_F1: .6f}")
            logging.info(f"{fold}折平均precise:{avg_precise: .6f}")
            logging.info(f"{fold}折平均recall:{avg_recall: .6f}")
        logging.info(f"--------------use all data to train--------------")
        is_train_all = True
        agent = MLE_Trainer(cfg, fold='all', train_idx=[i for i in data_idx], test_idx = [], is_train_all=True)
        for e in range(cfg['epoch']):
            agent.imitating(e+1, is_train_all)

    end = int(time.time())
    m, s = divmod(end - start, 60)
    print(f"Train model cost {m}min {s}s.")