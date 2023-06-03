# -*- coding: utf-8 -*-
import torch
import os
import json
from convlab2.policy.policy import Policy
from convlab2.policy.rlmodule import MultiDiscretePolicy
from convlab2.policy.vector.vector_diachat import DiachatVector
from convlab2.util.diachat.action_util import *
from diachatbot.ai.infer_impl import SimpleInference
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLE(Policy):

    def __init__(self):
        root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)

        sys_act_domain = os.path.join(root_dir, './data/diachat/sys_act_domain.json')
        usr_act_domain = os.path.join(root_dir, './data/diachat/usr_act_domain.json')
        sys_act_domain_slot = os.path.join(root_dir, './data/diachat/sys_act_domain_slot.json')
        usr_act_domain_slot = os.path.join(root_dir, './data/diachat/usr_act_domain_slot.json')
        self.vector = DiachatVector(sys_act_domain, usr_act_domain, sys_act_domain_slot, usr_act_domain_slot)

        self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.sys_da_dim).to(
            device=DEVICE)  # 三层模型

        self.load_from_pretrained(cfg['load'])
        self.simpleInfer = SimpleInference()

    def load_from_pretrained(self, filename):
        policy_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_mle.pol.mdl')
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))

    def format_action_arrays(self, actions):
        arry_actions = []
        for a in actions:
            ids = a.split('-')
            intent = ids[0]
            domain = ids[1] if ids[1] != 'none' else ''
            slot = ids[2] if ids[2] != 'none' else ''
            i_d_s = [intent, domain, slot]
            arry_actions.append(i_d_s)
        return arry_actions

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        a = self.policy.select_action(s_vec.to(device=DEVICE), False).cpu()
        action = self.vector.action_devectorize(a.detach().numpy())

        action = self.format_action_arrays(action)

        actionArry, anntActions = self.infer(state, action)

        # state['system_action'] = actionArry  # 单独评价或者测试policy的时候把这句注释掉
        return anntActions, action  # 第一个是调用推理后的，第二个是policy给出的

    def infer(self, state, policyActions):
        recmmdAction = self.simpleInfer.recommend_by_delex_da(state, policyActions)
        recmmdAction = self.add_default_action(recmmdAction)
        anntActions = recmmdAction_to_anntAction(recmmdAction)
        actionArry = anntAction_to_actionArry(anntActions)
        return actionArry, anntActions

    def add_default_action(self, recmmdAction):
        '''
                如果为空，也就是空数组[]，则添加默认的action，Chitchat
                添加的Chitchat为[[['Chitchat', '', '', '']]]
        '''
        if len(recmmdAction) == 0:
            recmmdAction.append([['Chitchat', '', '', '']])

        return recmmdAction

    def init_session(self):
        """
        Restore after one session
        """
        pass
