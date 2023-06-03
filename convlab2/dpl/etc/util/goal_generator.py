"""
generate usr goal from dataset annotations_goal.json
"""

import json
import os
import random
from pprint import pprint
from convlab2.dpl.vhus.diachat_DG_BiGRU.usermanager import UserDataManager


class GoalGenerator:
    def __init__(self, corpus_path='convlab2/dpl/etc/data/complete_data.json'):
        self.corpus_path = corpus_path

    def get_user_goal(self):
        sessions = json.load(open(self.corpus_path, encoding='UTF-8'))
        # 固定session 用于其它模块的测试
        # session = sessions[528]
        session = random.choice(sessions)
        goal = session.get('goal', [])
        user_goal = UserDataManager.change_goal_form(goal)
        return user_goal


if __name__ == '__main__':
    goal_generator = GoalGenerator()
    pprint(goal_generator.get_user_goal())
