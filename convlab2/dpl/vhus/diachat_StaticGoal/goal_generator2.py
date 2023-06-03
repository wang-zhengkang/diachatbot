"""
generate user goal from dataset annotations_goal.json
"""

import json
import os
import random
from pprint import pprint

from convlab2 import get_root_path

SORTS = {'current', 'AskForSure', 'AskFor', 'AskHow', 'AskWhy', 'Chitchat'}


class GoalGenerator:
    def __init__(self, corpus_path='data/diachat/annotations_goal.json'):
        self.corpus_path = corpus_path

    def get_user_goal(self):
        dialogs = json.load(open(self.corpus_path, encoding='UTF-8'))
        user_goal = dict([(sort, {}) for sort in SORTS])
        dialog = random.choice(dialogs)
        for _, goal in enumerate(dialog['goal']):
            for sort, sort_args in goal.items():
                if sort == 'current' or sort == 'Chitchat':
                    domain = sort_args[0]
                    slot = sort_args[1]
                    value = sort_args[2]
                    if domain not in user_goal[sort]:
                        user_goal[sort][domain] = {}
                    if slot not in user_goal[sort][domain]:
                        user_goal[sort][domain][slot] = []
                    if value not in user_goal[sort][domain][slot]:
                        user_goal[sort][domain][slot].append(value)
                elif sort == 'AskWhy':
                    why_list = sort_args['why']
                    if not why_list:
                        why_list = [['none', 'none', 'none']]
                    for why in why_list:
                        domain = why[0]
                        slot = why[1]
                        value = why[2]
                        if domain not in user_goal[sort]:
                            user_goal[sort][domain] = {}
                        if slot not in user_goal[sort][domain]:
                            user_goal[sort][domain][slot] = []
                        if value not in user_goal[sort][domain][slot]:
                            user_goal[sort][domain][slot].append(value)
                else:
                    arg_list = sort_args['args']
                    if not arg_list:
                        arg_list = [['', '', '']]
                    for arg in arg_list:
                        domain = arg[0]
                        slot = arg[1]
                        value = arg[2]
                        if domain not in user_goal[sort]:
                            user_goal[sort][domain] = {}
                        if slot not in user_goal[sort][domain]:
                            user_goal[sort][domain][slot] = []
                        if value not in user_goal[sort][domain][slot]:
                            user_goal[sort][domain][slot].append(value)
        return user_goal
    def get_user_list_form_goal(self):
        def change_goal_form(goals):
            user_goal = dict([(sort, []) for sort in SORTS])
            for goal in goals:
                for sort, sort_args in goal.items():
                    if sort == 'current':
                        domain = sort_args[0]
                        slot = sort_args[1]
                        value = sort_args[2]
                        temp_list = [domain + '-' + slot, value]
                        user_goal[sort].append(temp_list)
                    elif sort == 'AskWhy':
                        why_list = sort_args['why']
                        if not why_list:
                            user_goal[sort].append(['none-none', 'none'])
                            break
                        for why in why_list:
                            domain = why[0]
                            slot = why[1]
                            value = why[2]
                            temp_list = [domain + '-' + slot, value]
                            user_goal[sort].append(temp_list)
                    elif sort == 'Chitchat':
                        user_goal[sort].append(['none-none', 'none'])
                    else:
                        arg_list = sort_args['args']
                        if arg_list == [[]] or arg_list == [["", "", ""]]:
                            user_goal[sort].append(['none-none', 'none'])
                            break
                        for arg in arg_list:
                            domain = arg[0]
                            slot = arg[1]
                            value = arg[2]
                            temp_list = [domain + '-' + slot, value]
                            user_goal[sort].append(temp_list)
            return user_goal
        dialogs = json.load(open(self.corpus_path, encoding='UTF-8'))
        user_goal = dict([(sort, {}) for sort in SORTS])
        dialog = random.choice(dialogs)
        user_goal = change_goal_form(dialog.get('goal', []))
        return user_goal



if __name__ == '__main__':
    goal_generator = GoalGenerator()
    pprint(goal_generator.get_user_goal())
