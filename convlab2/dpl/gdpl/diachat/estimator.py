import os
import logging
import json
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from convlab2.util.train_util import to_device

from convlab2.dpl.etc.util.vector_diachat import DiachatVector
from convlab2.dpl.etc.loader.estimator_dataloader import EstimatorDataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardEstimator(object):
    def __init__(self, pretrain=False):
        with open('convlab2/dpl/gdpl/diachat/config.json', 'r') as f:
            cfg = json.load(f)
        vector = DiachatVector()
        self.irl = AIRL(cfg['gamma'], cfg['hi_dim'], vector.state_dim, vector.sys_da_dim).to(device=DEVICE)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.step = 0
        self.anneal = cfg['anneal']
        self.irl_params = self.irl.parameters()
        self.irl_optim = optim.RMSprop(self.irl_params, lr=cfg['lr_irl'])
        self.weight_cliping_limit = cfg['clip']

        self.save_dir = cfg['save_dir']
        self.save_per_epoch = cfg['save_per_epoch']
        self.optim_batchsz = cfg['batchsz']
        self.irl.eval()

        manager = EstimatorDataLoader()
        
        if pretrain:
            self.data_train = manager.create_dataset_irl('train', cfg['batchsz'])
            self.data_valid = manager.create_dataset_irl('val', cfg['batchsz'])
            self.data_test = manager.create_dataset_irl('test', cfg['batchsz'])
            self.irl_iter = iter(self.data_train)
            self.irl_iter_valid = iter(self.data_valid)
            self.irl_iter_test = iter(self.data_test)


    def kl_divergence(self, mu, logvar, istrain):
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        beta = min(self.step / self.anneal, 1) if istrain else 1
        return beta * klds

    def irl_loop(self, data_real, data_gen):
        s_real, a_real, next_s_real = to_device(data_real)
        s, a, next_s = data_gen

        # train with real data
        weight_real = self.irl(s_real, a_real, next_s_real)
        loss_real = -weight_real.mean()

        # train with generated data
        weight = self.irl(s, a, next_s)
        loss_gen = weight.mean()
        return loss_real, loss_gen

    def train_irl(self, batch, epoch):
        self.irl.train()
        input_s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        input_a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        input_next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        batchsz = input_s.size(0)

        real_loss, gen_loss = 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)

        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            try:
                data = next(self.irl_iter)
            except StopIteration:
                self.irl_iter = iter(self.data_train)
                data = next(self.irl_iter)

            self.irl_optim.zero_grad()
            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            loss = loss_real + loss_gen
            loss.backward()
            self.irl_optim.step()

            for p in self.irl_params:
                p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

        real_loss /= turns
        gen_loss /= turns
        logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}'.format(
            epoch, real_loss, gen_loss))
        if epoch % self.save_per_epoch == 0:
            self.save_irl(self.save_dir, epoch)
        self.irl.eval()

    def test_irl(self, batch, epoch, best):
        input_s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        input_a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        input_next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        batchsz = input_s.size(0)

        real_loss, gen_loss = 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)

        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            try:
                data = next(self.irl_iter_valid)
            except StopIteration:
                self.irl_iter_valid = iter(self.data_valid)
                data = next(self.irl_iter_valid)

            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()

        real_loss /= turns
        gen_loss /= turns
        logging.debug('<<reward estimator>> validation, epoch {}, loss_real:{}, loss_gen:{}'.format(
            epoch, real_loss, gen_loss))
        loss = real_loss + gen_loss
        if loss < best:
            logging.info('<<reward estimator>> best model saved')
            best = loss
            self.save_irl(self.save_dir, 'best')

        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            try:
                data = next(self.irl_iter_test)
            except StopIteration:
                self.irl_iter_test = iter(self.data_test)
                data = next(self.irl_iter_test)

            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()

        real_loss /= turns
        gen_loss /= turns
        logging.debug('<<reward estimator>> test, epoch {}, loss_real:{}, loss_gen:{}'.format(
            epoch, real_loss, gen_loss))
        return best

    def update_irl(self, inputs, batchsz, epoch):
        """
        train the reward estimator (together with encoder) using cross entropy loss (real, mixed, generated)
        Args:
            inputs: (s, a, next_s)
        """
        input_s, input_a, input_next_s = inputs

        real_loss, gen_loss = 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)

        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            try:
                data = next(self.irl_iter)
            except StopIteration:
                self.irl_iter = iter(self.data_train)
                data = next(self.irl_iter)

            self.irl_optim.zero_grad()
            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            loss = loss_real + loss_gen
            loss.backward()
            self.irl_optim.step()

            for p in self.irl_params:
                p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

        real_loss /= turns
        gen_loss /= turns
        logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}'.format(
            epoch, real_loss, gen_loss))

        if (epoch+1) % self.save_per_epoch == 0:
            self.save_irl(self.save_dir, epoch)

    def save_irl(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.irl.state_dict(), directory + '/' + str(epoch) + '_estimator.mdl')
        logging.info('<<reward estimator>> epoch {}: saved network to mdl'.format(epoch))

    def load_irl(self, filename):
        irl_mdl = filename + '_estimator.mdl'
        if os.path.exists(irl_mdl):
            self.irl.load_state_dict(torch.load(irl_mdl, map_location=DEVICE))
            logging.info('<<reward estimator>> loaded checkpoint from file: {}'.format(irl_mdl))

    def estimate(self, s, a, next_s, log_pi):
        """
        infer the reward of state action pair with the estimator
        """
        weight = self.irl(s, a.float(), next_s)
        logging.debug('<<reward estimator>> weight {}'.format(weight.mean().item()))
        logging.debug('<<reward estimator>> log pi {}'.format(log_pi.mean().item()))
        # see AIRL paper
        # r = f(s, a, s') - log_p(a|s)
        reward = (weight - log_pi).squeeze(-1)
        return reward


class AIRL(nn.Module):
    """
    label: 1 for real, 0 for generated
    """

    def __init__(self, gamma, h_dim, s_dim, a_dim):
        super(AIRL, self).__init__()

        self.gamma = gamma
        self.g = nn.Sequential(nn.Linear(s_dim + a_dim, h_dim),
                               nn.ReLU(),
                               nn.Linear(h_dim, 1))
        self.h = nn.Sequential(nn.Linear(s_dim, h_dim),
                               nn.ReLU(),
                               nn.Linear(h_dim, 1))

    def forward(self, s, a, next_s):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :param next_s: [b, s_dim]
        :return:  [b, 1]
        """
        weights = self.g(torch.cat([s, a], -1)) + self.gamma * self.h(next_s) - self.h(s)
        return weights


# class ActEstimatorDataLoaderDiachat(ActMLEPolicyDataLoaderDiachat):
#     def __init__(self):
#         super(ActEstimatorDataLoaderDiachat, self).__init__()

#     def create_dataset_irl(self, part, batchsz):
#         print('Start creating {} irl dataset'.format(part))
#         s = []
#         a = []
#         next_s = []
#         for i, item in enumerate(self.data[part]):
#             s.append(torch.Tensor(item[0]))
#             a.append(torch.Tensor(item[1]))
#             if item[0][-1]:  # terminated
#                 next_s.append(torch.Tensor(item[0]))
#             else:
#                 next_s.append(torch.Tensor(self.data[part][i + 1][0]))
#         s = torch.stack(s)
#         a = torch.stack(a)
#         next_s = torch.stack(next_s)
#         dataset = ActStateDataset(s, a, next_s)
#         dataloader = data.DataLoader(dataset, batchsz, True)
#         print('Finish creating {} irl dataset'.format(part))
#         return dataloader