import os
import torch
import logging
import torch.nn as nn
import json
import time
import zipfile

from sklearn.model_selection import KFold
from convlab2.util.train_util import to_device, init_logging_handler

from convlab2.dpl.rlmodule import MultiDiscretePolicy
from convlab2.dpl.etc.util.vector_diachat import DiachatVector
from convlab2.dpl.etc.util.split_list import split_list
from convlab2.dpl.etc.loader.policy_dataloader import PolicyDataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skf = KFold(n_splits=10, shuffle=True, random_state=2023)
skf1 = KFold(n_splits=9, shuffle=True, random_state=2023)


class MLE_Trainer():

    def __init__(self, cfg, fold, train_idx, test_idx, is_train_all=False):

        vector = DiachatVector()

        self.save_dir = cfg['save_dir']
        self.print_per_batch = cfg['print_per_batch']
        self.save_per_epoch = cfg['save_per_epoch']
        self.batchsz = cfg['batchsz']
        self.epoch = cfg['epoch']
        self.lr = cfg['lr']
        self.h_dim = cfg['h_dim']
        
        self.fold = fold
        self.kfold_info = {
            'F1': float,
            'precise': float,
            'recall': float
        }

        self.manager = PolicyDataloader()
        self.data_train = self.manager.create_dataset('train', cfg['batchsz'], train_idx)
        
        # 如果不使用全部数据训练 则创建测试集
        if not is_train_all:
            self.data_test = self.manager.create_dataset('test', cfg['batchsz'], test_idx)

        self.policy = MultiDiscretePolicy(vector.state_dim, cfg['h_dim'], vector.sys_da_dim).to(
            device=DEVICE)
        self.policy.eval()

        # torch.optim指定以adam作为优化器
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg['lr'])
        # 指定损失函数为多标签分类损失
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()

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

            if not is_train_all:
                if (i + 1) % self.print_per_batch == 0:
                    a_loss /= self.print_per_batch
                    logging.debug('<<DPL-MLE>> epoch {}, iter {}, loss_a:{}'.format(epoch, i, a_loss))
                    a_loss = 0.

        if epoch == self.epoch:
            self.save(self.save_dir, is_train_all)  # 以.mdl保存模型
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
        if not os.path.exists(directory):
                os.makedirs(directory)

        if is_train_all:
            print("sava train_all_mle.pol.mdl")
            torch.save(self.policy.state_dict(), directory + '/' + 'train_all' + '_mle.pol.mdl')
        else:
            torch.save(self.policy.state_dict(), directory + '/' + str(self.fold) + '_fold_mle.pol.mdl')

            logging.info(
                '<<DPL-MLE>> saved {} fold network to mdl'.format(self.fold))

    def load(self, filename='save/best'):
        policy_mdl = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), filename + '_mle.pol.mdl')
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))


if __name__ == '__main__':
    with open('convlab2/dpl/mle/diachat/config.json', 'r') as f:
        cfg = json.load(f)
    with open('convlab2/dpl/etc/data/complete_data.json', 'r') as fp:
        data = json.load(fp)
        data_idx = range(len(data))

    init_logging_handler(cfg['log_dir'])
    batchsz = cfg['batchsz']
    epoch = cfg['epoch']
    lr = cfg['lr']
    h_dim = cfg['h_dim']


    total_F1 = 0.
    total_precise = 0.
    total_recall = 0.
    best_F1 = 0.
    worst_F1 = 0.
    best_F1_fold = 1
    worst_F1_fold = 1

    start = int(time.time())

    for fold, (train_idx, test_idx) in enumerate(skf.split(data_idx)):
        fold += 1
        
        agent = MLE_Trainer(cfg, fold, train_idx, test_idx)
        logging.info(f"--------------第{fold}折交叉训练--------------")
        for e in range(cfg['epoch']):
            agent.imitating(e+1)
        agent.test()
        if fold == 1:
            best_F1 = agent.kfold_info['F1']
            worst_F1 = agent.kfold_info['F1']
        else:
            if best_F1 < agent.kfold_info['F1']:
                best_F1 = agent.kfold_info['F1']
                best_F1_fold = fold
            if worst_F1 > agent.kfold_info['F1']:
                worst_F1 = agent.kfold_info['F1']
                worst_F1_fold = fold
        total_F1 += agent.kfold_info['F1']
        total_precise += agent.kfold_info['precise']
        total_recall += agent.kfold_info['recall']
        avg_F1 = total_F1 / fold
        avg_precise = total_precise / fold
        avg_recall = total_recall / fold
        logging.info(f"{fold}折平均F1:{avg_F1: .6f}")
        logging.info(f"{fold}折平均precise:{avg_precise: .6f}")
        logging.info(f"{fold}折平均recall:{avg_recall: .6f}")

    is_train_all = True  # 是否使用全部数据进行训练
    if is_train_all:
        print(f"\nuse all data to train: pls waiting...")
        agent = MLE_Trainer(cfg, fold='all', train_idx=[i for i in data_idx],
                            test_idx=[], is_train_all=True)
        for e in range(cfg['epoch']):
            agent.imitating(e+1, is_train_all)
        print("use all data to train: complete")
    
    end = int(time.time())
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)

    logging.info(f"Train model cost time {h:0>2d}:{m:0>2d}:{s:0>2d}.")
    logging.info(f"--------------Statistics info start---------------")
    logging.info(f'batchsz:{batchsz}')
    logging.info(f'epoch:{epoch}')
    logging.info(f'lr:{lr}')
    logging.info(f'h_dim:{h_dim}')
    logging.info(f'random_state:2023')
    logging.info(f"avg F1:{avg_F1: .6f}")
    logging.info(f"avg precise:{total_precise/10: .6f}")
    logging.info(f"avg recall:{total_recall/10: .6f}")
    logging.info(f"best F1:{best_F1: .6f}, best F1 fold:{best_F1_fold}")
    logging.info(f"worst F1:{worst_F1: .6f}, worst F1 fold:{worst_F1_fold}")
    logging.info(f"--------------Statistics info finish--------------")
    
    for fold, (train_idx_, test_idx) in enumerate(skf.split(data_idx)):
        fold += 1
        if fold == best_F1_fold:
            for _, (train_idx, val_idx) in enumerate(skf1.split(train_idx_)):
                val_idx = val_idx
                logging.info(f"构建best fold:{best_F1_fold}数据集")
                break
            dataname = ['test', 'val', 'train']
            splited_data = {
                "train_data": [data[i] for i in train_idx],
                "val_data": [data[i] for i in val_idx],
                "test_data": [data[i] for i in test_idx]
            }
            for name in dataname:
                print('{}数据集数据量为：{}'.format(name, len(splited_data['{}_data'.format(name)])))

                split_json_file = f"convlab2/dpl/etc/data/{name}.json"
                with open(split_json_file, 'w', encoding='utf-8') as f:
                    json.dump(splited_data['{}_data'.format(name)], f, ensure_ascii=False, sort_keys=True, indent=4)

                f = zipfile.ZipFile(split_json_file + '.zip', 'w', zipfile.ZIP_DEFLATED)
                f.write(split_json_file, arcname='{}.json'.format(name))
                f.close()
