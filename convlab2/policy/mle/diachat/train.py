import os
import torch
import logging
import torch.nn as nn
import json
import pickle
import sys
import random
import numpy as np

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

from convlab2.policy.rlmodule import MultiDiscretePolicy
from convlab2.policy.vector.vector_diachat import DiachatVector
from convlab2.policy.mle.diachat.loader import PolicyDataLoaderDiachat
from convlab2.util.train_util import to_device, init_logging_handler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLE_Trainer():
    def __init__(self, cfg):
        sys_ad = 'convlab2/policy/mle/diachat/processed_data/sys_ad.json'
        user_ad = 'convlab2/policy/mle/diachat/processed_data/user_ad.json'
        sys_ads = 'convlab2/policy/mle/diachat/processed_data/sys_ads.json'
        user_ads = 'convlab2/policy/mle/diachat/processed_data/user_ads.json'
        vector = DiachatVector(sys_ad, user_ad, sys_ads, user_ads)

        self.manager = PolicyDataLoaderDiachat(vector)

        self.data_train = self.manager.create_dataset('train', cfg['batchsz'])
        self.data_valid = self.manager.create_dataset('val', cfg['batchsz'])
        self.data_test = self.manager.create_dataset('test', cfg['batchsz'])
        self.save_dir = cfg['save_dir']
        self.print_per_batch = cfg['print_per_batch']
        self.save_per_epoch = cfg['save_per_epoch']

        self.policy = MultiDiscretePolicy(vector.state_dim, cfg['h_dim'], vector.sys_da_dim).to(
            device=DEVICE)  # 构建一个神经网络，入度为328，出度为155
        self.policy.eval()
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg['lr'])  # torch.optim指定以adam作为优化器
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()  # 指定损失函数为多标签分类损失

    def policy_loop(self, data):
        s, target_a = to_device(data)
        a_weights = self.policy(s)  # 进行一次训练，得到a_weights

        loss_a = self.multi_entropy_loss(a_weights, target_a)  # 计算损失
        return loss_a

    def imitating(self, epoch):
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
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_a:{}'.format(epoch, i, a_loss))
                a_loss = 0.

        if (epoch + 1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)  # 保存一下训练的模型，以mdl模式
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
        logging.debug('<<dialog policy>> validation, epoch {}, loss_a:{}'.format(epoch, a_loss))
        if a_loss < best:
            logging.info('<<dialog policy>> best model saved')
            best = a_loss
            self.save(self.save_dir, 'best')

        a_loss = 0.
        for i, data in enumerate(self.data_test):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()

        a_loss /= len(self.data_test)
        logging.debug('<<dialog policy>> test, epoch {}, loss_a:{}'.format(epoch, a_loss))
        return best

    def test(self):
        def f1(a, target):
            TP, FP, FN = 0, 0, 0
            real = target.nonzero().tolist()
            predict = a.nonzero().tolist()  # 返回不为0的元素下标，全0是什么鬼，torch.Size([50, 331])
            # print(real)
            # print(predict)
            # print()
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
            s, target_a = to_device(data)  # s=torch.Size([14, 813])),target_a=torch.Size([14, 331]))
            a_weights = self.policy(s)  # a_weights=torch.Size([14, 331])
            a = a_weights.ge(0)  # 关系运算符，大于等于的意思
            # TODO: fix batch F1
            TP, FP, FN = f1(a, target_a)
            a_TP += TP
            a_FP += FP
            a_FN += FN

        prec = a_TP / (a_TP + a_FP)
        rec = a_TP / (a_TP + a_FN)
        F1 = 2 * prec * rec / (prec + rec)
        logging.info(f'a_TP: {a_TP}, a_FP: {a_FP}, a_FN: {a_FN}, F1: {F1}')

    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_mle.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

    def load(self, filename='save/best'):
        policy_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_mle.pol.mdl')
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))


if __name__ == '__main__':
    # random_seed = 2019
    # random.seed(random_seed)
    # np.random.seed(random_seed)
    # torch.manual_seed(random_seed)  # 用于设计随机初始化种子，保证初始化每次都相同
    # manager = PolicyDataLoaderDiachat()

    with open('convlab2/policy/mle/diachat/config.json', 'r') as f:
        cfg = json.load(f)
    init_logging_handler(cfg['log_dir'])
    agent = MLE_Trainer(cfg)
    agent.load()

    logging.debug('start training')

    best = float('inf')
    for e in range(cfg['epoch']):
        agent.imitating(e)
        best = agent.imit_test(e, best)
    agent.test()  # 5731 1483 1880 0.7731534569983137
