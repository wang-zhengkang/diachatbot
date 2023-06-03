# -*- coding: utf-8 -*-
"""
@author: truthless
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn

from convlab2.util.train_util import to_device
from convlab2.policy.vhus.diachat_DG_BiGRU.usermodule import VHUS
from convlab2.policy.vhus.diachat_DG_BiGRU.util import padding_data, kl_gaussian

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    DEVICES = [0, 1]
    # DEVICES = [0]

def batch_iter(x, y, z, batch_size=64):
    data_len = len(x)
    num_batch = ((data_len - 1) // batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = np.array(x)[indices]
    y_shuffle = np.array(y)[indices]
    z_shuffle = np.array(z)[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], z_shuffle[start_id:end_id]


class VHUS_Trainer():
    def __init__(self, config, manager, goal_gen, fold, train_idx, test_idx):
        
        voc_goal_size, voc_usr_size, voc_sys_size = manager.get_voc_size()
        self.user = VHUS(config, voc_goal_size, voc_usr_size, voc_sys_size)
        self.user = nn.DataParallel(self.user, device_ids=DEVICES).to(device=device)
        self.goal_gen = goal_gen
        self.manager = manager
        self.print_per_batch = config['print_per_batch']
        self.save_dir = config['save_dir']
        self.save_per_epoch = config['save_per_epoch']
        self.fold = fold
        self.kfold_info = {
            'F1': float,
            'precise': float,
            'recall': float
        }
        seq_goalss, seq_usr_dass, seq_sys_dass = manager.data_loader_seg()
        
        train_goals, train_usrdas, train_sysdas, test_goals, test_usrdas, test_sysdas = \
            manager.kfold_date_process(seq_goalss, seq_usr_dass, seq_sys_dass, train_idx, test_idx)
        self.data_train = (train_goals, train_usrdas, train_sysdas, config['batchsz'])
        # self.data_valid = (val_goals, val_usrdas, val_sysdas, config['batchsz'])
        self.data_test = (test_goals, test_usrdas, test_sysdas, config['batchsz'])
        
        self.alpha = config['alpha']
        self.optim = torch.optim.Adam(self.user.module.parameters(), lr=config['lr'])
        self.nll_loss = nn.NLLLoss(ignore_index=0).cuda()  # PAD=0
        self.bce_loss = nn.BCEWithLogitsLoss().cuda()


    def user_loop(self, data):
        batch_input = to_device(padding_data(data))
        a_weights, t_weights, argu = self.user(batch_input['goals'], batch_input['goals_length'], \
                                               batch_input['posts'], batch_input['posts_length'],
                                               batch_input['origin_responses'])

        loss_a, targets_a = 0, batch_input['origin_responses'][:, 1:]  # remove sos_id
        for i, a_weight in enumerate(a_weights):
            loss_a += self.nll_loss(a_weight, targets_a[:, i])
        loss_a /= i
        loss_t = self.bce_loss(t_weights, batch_input['terminated'])
        loss_a += self.alpha * kl_gaussian(argu)
        return loss_a, loss_t

    def imitating(self, epoch, is_train_all=False):
        """
        train the user simulator by simple imitation learning (behavioral cloning)
        """
        self.user.train()
        a_loss, t_loss = 0., 0.
        data_train_iter = batch_iter(self.data_train[0], self.data_train[1], self.data_train[2], self.data_train[3])
        for i, data in enumerate(data_train_iter):
            self.optim.zero_grad()
            loss_a, loss_t = self.user_loop(data)
            a_loss += loss_a.item()
            t_loss += loss_t.item()
            loss = loss_a + loss_t
            loss.backward()
            self.optim.step()

            if not is_train_all:
                if (i + 1) % self.print_per_batch == 0:
                    a_loss /= self.print_per_batch
                    t_loss /= self.print_per_batch
                    logging.debug(
                        '<<US>> fold {}, epoch {}, iter {}, loss_a:{}, loss_t:{}'.format(self.fold, epoch, i, a_loss, t_loss))
                    a_loss, t_loss = 0., 0.
        # if (epoch + 1) % self.save_per_epoch == 0:
        #     self.save(self.save_dir, epoch)
        
        if is_train_all:
            self.save(self.save_dir, self.fold, is_train_all)
        
        self.user.module.eval()

    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the user simulator fit on the training dataset
        """
        a_loss, t_loss = 0., 0.
        data_valid_iter = batch_iter(self.data_test[0], self.data_test[1], self.data_test[2], self.data_test[3])
        for i, data in enumerate(data_valid_iter):
            loss_a, loss_t = self.user_loop(data)
            a_loss += loss_a.item()
            t_loss += loss_t.item()

        a_loss /= i + 1
        t_loss /= i + 1
        # logging.debug('<<US>> val, fold {}, epoch {}, loss_a:{}, loss_t:{}'.format(self.fold, epoch, a_loss, t_loss))
        loss = a_loss + t_loss
        if loss < best:
            # logging.info(f'<<US>> fold best model saved')
            best = loss
            self.save(self.save_dir, f'{self.fold}')

        # a_loss, t_loss = 0., 0.
        # data_test_iter = batch_iter(self.data_test[0], self.data_test[1], self.data_test[2], self.data_test[3])
        # for i, data in enumerate(data_test_iter):
        #     loss_a, loss_t = self.user_loop(data)
        #     a_loss += loss_a.item()
        #     t_loss += loss_t.item()

        # a_loss /= i + 1
        # t_loss /= i + 1
        # logging.debug('<<user simulator>> test, epoch {}, loss_a:{}, loss_t:{}'.format(epoch, a_loss, t_loss))
        return best

    def test(self):
        def sequential(da_seq):
            # ads
            das = []
            for da in da_seq:
                das.append(da)
            return das
            
            # a-ds
            # das = []
            # cur_act = None
            # for word in da_seq:
            #     if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '(', ')']:
            #         continue
            #     if '-' not in word:
            #         cur_act = word
            #     else:
            #         if cur_act is None:
            #             continue
            #         das.append(cur_act + '-' + word)
            # return das

        def f1(pred, real):
            if not real:
                return 0, 0, 0
            TP, FP, FN = 0, 0, 0
            for item in real:
                if item in pred:
                    TP += 1
                else:
                    FN += 1
            for item in pred:
                if item not in real:
                    FP += 1
            return TP, FP, FN

        data_test_iter = batch_iter(self.data_test[0], self.data_test[1], self.data_test[2], self.data_test[3])
        a_TP, a_FP, a_FN, t_corr, t_tot = 0, 0, 0, 0, 0
        eos_id = self.user.module.usr_decoder.eos_id
        for i, data in enumerate(data_test_iter):
            batch_input = to_device(padding_data(data))
            a_weights, t_weights, argu = self.user.module(batch_input['goals'], batch_input['goals_length'], \
                                                   batch_input['posts'], batch_input['posts_length'],
                                                   batch_input['origin_responses'])
            usr_a = []
            for a_weight in a_weights:
                usr_a.append(a_weight.argmax(1).cpu().numpy())
            usr_a = np.array(usr_a).T.tolist()
            a = []
            for ua in usr_a:
                if eos_id in ua:
                    ua = ua[:ua.index(eos_id)]
                a.append(sequential(self.manager.id2sentence(ua)))
            targets_a = []
            for ua_sess in data[1]:
                for ua in ua_sess:
                    targets_a.append(sequential(self.manager.id2sentence(ua[1:-1])))
            TP, FP, FN = f1(a, targets_a)
            a_TP += TP
            a_FP += FP
            a_FN += FN

            t = t_weights.ge(0).cpu().tolist()
            targets_t = batch_input['terminated'].cpu().long().tolist()
            judge = np.array(t) == np.array(targets_t)
            t_corr += judge.sum()
            t_tot += judge.size

        precise = a_TP / (a_TP + a_FP)
        recall = a_TP / (a_TP + a_FN)
        F1 = 2 * precise * recall / (precise + recall + 0.0001)
        self.kfold_info['F1'] = F1
        self.kfold_info['precise'] = precise
        self.kfold_info['recall'] = recall
        logging.info(f'a_TP={a_TP}, a_FP={a_FP}, a_FN={a_FN}, F1={F1: .6f}')
        logging.info(f'precise={precise: .6f}, recall={recall: .6f}')
        logging.info(f't_corr={t_corr}, t_tot={t_tot}, t_corr / t_tot={t_corr / t_tot: .4f}')
        self.save(self.save_dir, self.fold)

    def save(self, directory, fold, is_train_all=False):
        if is_train_all:
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(self.user.module.state_dict(), directory + '/' + f'{fold}' + '_data_simulator.mdl')
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(self.user.module.state_dict(), directory + '/' + str(fold) + '_fold_best_simulator.mdl')
            logging.info('saved {} fold best network to mdl'.format(fold))
        