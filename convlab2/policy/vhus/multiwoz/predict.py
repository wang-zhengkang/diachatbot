# -*- coding: utf-8 -*-
import os
import json
import sys
from pprint import pprint

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

from convlab2.task.multiwoz.goal_generator import GoalGenerator
from convlab2.policy.vhus.multiwoz.usermanager import UserDataManager
from convlab2.policy.vhus.multiwoz.vhus import UserPolicyVHUS


if __name__ == '__main__':
    goal_gen = GoalGenerator()
    user = UserPolicyVHUS()
    user.init_session()
    pprint(user.goal)
    pprint(user.predict([['<PAD>', '<PAD>', '<PAD>', '<PAD>']]))
