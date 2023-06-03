"""
进行goal构造，以及重新构造user_state
"""
import json
import os
import copy
from collections import defaultdict


def getCategory(act_label):
    if act_label == "Inform":
        return "现状"
    if act_label in ["Advice", "AdviceNot"]:
        return "建议"
    else:
        return "解释"


current_dir = os.path.dirname(os.path.abspath(__file__))  # ..\diachatbot\data\diachat
data_source_name = "annotations_nothanks.json"
target_file_name = "annotations_goal.json"
data_source_dir = os.path.join(current_dir, data_source_name)
target_file_dir = os.path.join(current_dir, target_file_name)

with open(data_source_dir, encoding="utf-8") as data_source:
    with open(target_file_dir, mode='w', encoding='utf-8') as target_file:
        # 加载数据
        data_source = json.load(data_source)
        # 处理完毕的数据
        processed_data = []
        # goal_file.write(json.dumps(data_source, ensure_ascii=False))

        # 遍历每一段对话
        for _, conversation in enumerate(data_source):
            # init user_state, sys_state, sys_state_init, goal
            user_state = []
            sys_state_init = {}
            sys_state = {}


            # 存储上一个角色的annotation
            user_annotation = []
            sys_annotation = []

            # domains
            domains = []

            if "agentId" in conversation.keys():
                conversation = {
                    'conversationId': conversation['conversationId'],
                    'type': conversation['type'],
                    'domains': domains,
                    'agentId': conversation['agentId'],
                    'goal': [],
                    'utterances': conversation['utterances'],
                    'final_goal': []
                }
            else:
                conversation = {
                    'conversationId': conversation['conversationId'],
                    'type': conversation['type'],
                    'domains': domains,
                    'goal': [],
                    'utterances': conversation['utterances'],
                    'final_goal': []
                }
            # 遍历每句话 goal初始化
            for i, utterance in enumerate(conversation['utterances']):
                # goal初始化
                for _, annotation in enumerate(utterance['annotation']):
                    if utterance['agentRole'] == 'User' and annotation['act_label'] == 'Inform':
                        for s in annotation['slot_values']:
                            goal_temp = [s["domain"], s["slot"], s["value"], False]
                            conversation['goal'].append(
                                {
                                    "current": goal_temp
                                }
                            )
                    if utterance['agentRole'] == 'User' and annotation['act_label'] == 'AskFor':
                        args = []
                        AskFor = {}
                        for s in annotation['slot_values']:
                            args.append([s["domain"], s["slot"], s["value"]])
                        AskFor['args'] = args
                        AskFor['done'] = False
                        conversation['goal'].append(
                            {
                                "AskFor": AskFor
                            }
                        )
                    if utterance['agentRole'] == 'User' and annotation['act_label'] == 'AskForSure':
                        args = []
                        AskForSure = {}
                        for s in annotation['slot_values']:
                            args.append([s["domain"], s["slot"], s["value"]])
                        AskForSure['args'] = args
                        AskForSure['done'] = False
                        AskForSure['sure'] = ""
                        conversation['goal'].append(
                            {
                                "AskForSure": AskForSure
                            }
                        )
                    if utterance['agentRole'] == 'User' and annotation['act_label'] == 'AskHow':
                        args = []
                        AskHow = {}
                        for s in annotation['slot_values']:
                            args.append([s["domain"], s["slot"], s["value"]])
                        AskHow['args'] = args
                        AskHow['done'] = False
                        AskHow['how'] = []
                        conversation['goal'].append(
                            {
                                "AskHow": AskHow
                            }
                        )
                    if utterance['agentRole'] == 'User' and annotation['act_label'] == 'AskWhy':
                        AskWhy = {'done': False, 'why': []}
                        conversation['goal'].append(
                            {
                                "AskWhy": AskWhy
                            }
                        )
                    if utterance['agentRole'] == 'User' and annotation['act_label'] == 'Chitchat' and annotation:
                        conversation['goal'].append(
                            {
                                "Chitchat": ['none', 'none', 'none', False]
                            }
                        )
            user_state = copy.deepcopy(conversation['goal'])
            goal_count = 0  # goal条数计数
            goal_count2 = {}  # 用于记录AskFor, AskForSure等act处在goal的位置,便于goal中以上act的更新。"goal_count": "act"
            # 遍历每一句话
            for i, utterance in enumerate(conversation['utterances']):
                # sys的每一句话中sys_state_init, sys_state只更新一次
                sys_state_init_flag = True
                sys_state_flag = True
                # goal, user_state更新
                for _, annotation in enumerate(utterance['annotation']):
                    if utterance['agentRole'] == 'User':
                        if annotation['act_label'] == 'Inform':
                            # 需要判断一个ACT多个SV的情况
                            for _ in range(0, len(annotation['slot_values'])):
                                user_state[goal_count]['current'][3] = True
                                goal_count += 1
                            continue
                        if annotation['act_label'] == 'AskFor':
                            user_state[goal_count]['AskFor']["done"] = True
                            goal_count2.update({goal_count: annotation['act_label']})
                            goal_count += 1
                            continue
                        if annotation['act_label'] == 'AskForSure':
                            user_state[goal_count]['AskForSure']["done"] = True
                            goal_count2.update({goal_count: annotation['act_label']})
                            goal_count += 1
                            continue
                        if annotation['act_label'] == 'AskHow':
                            user_state[goal_count]['AskHow']["done"] = True
                            goal_count2.update({goal_count: annotation['act_label']})
                            goal_count += 1
                            continue
                        if annotation['act_label'] == 'AskWhy':
                            user_state[goal_count]['AskWhy']["done"] = True
                            goal_count2.update({goal_count: annotation['act_label']})
                            goal_count += 1
                            continue
                        if annotation['act_label'] == 'Chitchat':
                            user_state[goal_count]['Chitchat'][3] = True
                            goal_count2.update({goal_count: annotation['act_label']})
                            goal_count += 1
                            continue
                    if utterance['agentRole'] == 'Doctor':
                        for key, value in goal_count2.items():  # key是goal_count, value是act
                            if value == "AskFor":
                                if annotation['act_label'] == 'Advice':
                                    user_state[key]['AskFor']['args'][-1][-1] = annotation['slot_values'][-1]["value"]
                            elif value == "AskForSure":
                                if annotation['act_label'] in ['Assure', 'Deny']:
                                    user_state[key]['AskForSure']['sure'] = annotation['act_label']
                            elif value == "AskHow":
                                if annotation['act_label'] == 'Advice':
                                    for sv in annotation['slot_values']:
                                        temp = [sv['domain'], sv['slot'], sv['value']]
                                        user_state[key]['AskHow']['how'].append(temp)
                            elif value == "AskWhy":
                                if annotation['act_label'] == 'Explanation':
                                    for sv in annotation['slot_values']:
                                        temp = [sv['domain'], sv['slot'], sv['value']]
                                        user_state[key]['AskWhy']['why'].append(temp)
                    goal_count2 = {}  # 重置goal_count2
                for _, annotation in enumerate(utterance['annotation']):
                    # sys_state_init: sys_state, user_annotation
                    if sys_state_init_flag:
                        sys_state_init = copy.deepcopy(sys_state)
                        sys_state_init_flag = False
                    if utterance['agentRole'] == 'User' and annotation['act_label'] in ['Inform', 'Advice', 'AdviceNot',
                                                                                        'Explanation']:
                        sys_state_init_temp = {}
                        for s in annotation['slot_values']:
                            category = getCategory(annotation['act_label'])

                            # 判断有无一样的key值，如有则分开放。没有，则将同一act_label下的slot-value放一起
                            try:
                                sys_state_init_temp[s['domain']][category]
                            except:
                                sys_state_init_temp[s['domain']] = {
                                    category: []
                                }
                            if not sys_state_init_temp[s['domain']][category]:
                                sys_state_init_temp[s['domain']][category].append({s['slot']: s['value']})
                            else:
                                for dic in sys_state_init_temp[s['domain']][category]:
                                    key_list = []
                                    for key, value in dic.items():
                                        key_list.append(key)
                                for _ in range(len(key_list)):
                                    if s['slot'] in key_list:
                                        sys_state_init_temp[s['domain']][category].append({s['slot']: s['value']})
                                    else:
                                        sys_state_init_temp[s['domain']][category][0].update(
                                            {s['slot']: s['value']})
                        for domain, cat_dic in sys_state_init_temp.items():
                            for cat, _ in cat_dic.items():
                                try:
                                    sys_state_init[domain][cat]
                                except:
                                    try:
                                        sys_state_init[domain].update({
                                            cat: []
                                        })
                                    except:
                                        sys_state_init[domain] = {
                                            cat: []
                                        }
                                sys_state_init[domain][cat] += sys_state_init_temp[domain][cat]

                        sys_state_flag = True
                    # sys_state: sys_state_init, sys_annotation
                    if sys_state_flag:
                        sys_state = copy.deepcopy(sys_state_init)
                        sys_state_flag = False
                    if utterance['agentRole'] == 'Doctor' and annotation['act_label'] in ['Inform', 'Advice',
                                                                                          'AdviceNot', 'Explanation']:
                        sys_state_temp = {}
                        act_len = len(annotation['slot_values'])
                        for sv_num, s in enumerate(annotation['slot_values']):
                            num = 0
                            category = getCategory(annotation['act_label'])
                            try:
                                sys_state_temp[s['domain']][category]
                            except:
                                sys_state_temp[s['domain']] = {
                                    category: []
                                }
                            if not sys_state_temp[s['domain']][category]:
                                sys_state_temp[s['domain']][category].append({s['slot']: s['value']})
                            else:
                                for dic in sys_state_temp[s['domain']][category]:
                                    key_list = []
                                    for key, value in dic.items():
                                        if key != '是否建议':
                                            key_list.append(key)
                                for _ in range(len(key_list)):
                                    if s['slot'] in key_list:
                                        sys_state_temp[s['domain']][category].append({s['slot']: s['value']})
                                        num += 1

                                        if annotation['act_label'] == "Advice":
                                            sys_state_temp[s['domain']][category][num - 1].update({'是否建议': 'true'})
                                        elif annotation['act_label'] == "AdviceNot":
                                            sys_state_temp[s['domain']][category][num - 1].update({'是否建议': 'false'})
                                    else:
                                        sys_state_temp[s['domain']][category][num].update(
                                            {s['slot']: s['value']})
                            if sv_num + 1 == act_len:
                                if annotation['act_label'] == "Advice":
                                    sys_state_temp[s['domain']][category][-1].update({'是否建议': 'true'})
                                elif annotation['act_label'] == "AdviceNot":
                                    sys_state_temp[s['domain']][category][-1].update({'是否建议': 'false'})

                        for domain, cat_dic in sys_state_temp.items():
                            for cat, _ in cat_dic.items():
                                try:
                                    sys_state[domain][cat]
                                except:
                                    try:
                                        sys_state[domain].update({
                                            cat: []
                                        })
                                    except:
                                        sys_state[domain] = {
                                            cat: []
                                        }
                                sys_state[domain][cat] += sys_state_temp[domain][cat]

                if utterance['agentRole'] == 'User':
                    conversation['utterances'][i]['user_state'] = copy.deepcopy(user_state)
                else:
                    conversation['utterances'][i]['sys_state_init'] = copy.deepcopy(sys_state_init)
                    conversation['utterances'][i]['sys_state'] = copy.deepcopy(sys_state)

            # domain and domains
            for i, utterance in enumerate(conversation['utterances']):
                # conversation['domain']
                conversation['utterances'][i]['domain'] = ''
                domain_temp = []
                for _, annotation in enumerate(utterance['annotation']):
                    if annotation["act_label"] in ['Inform', 'Advice', 'AdviceNot', 'Explanation', 'AskFor', 'AskHow']:
                        for s in annotation["slot_values"]:
                            if s['domain'] not in domains and s['domain'] != '':
                                domains.append(s["domain"])
                            if s["domain"] not in domain_temp:
                                domain_temp.append(s["domain"])
                    if annotation["act_label"] in ['Accept', 'Assure', 'Deny', 'AskWhy', 'Uncertain',
                                                   'AskForSure'] and len(utterance['annotation']) == 1:
                        for s in annotation["slot_values"]:
                            if s['domain'] not in domains and s['domain'] != '':
                                domains.append(s["domain"])
                        domain_temp += before_domain_temp
                        domain_temp = list(set(domain_temp))
                    if annotation["act_label"] in ['GeneralAdvice', 'GeneralExplanation', 'Chitchat'] and len(
                            utterance['annotation']) == 1:
                        continue
                for domain in domain_temp:
                    if conversation['utterances'][i]['domain'] == '':
                        conversation['utterances'][i]['domain'] = domain
                    else:
                        conversation['utterances'][i]['domain'] = conversation['utterances'][i][
                                                                      'domain'] + '-' + domain
                before_domain_temp = domain_temp

            conversation['domains'] = domains
            conversation['final_goal'] = user_state
            processed_data.append(conversation)
        target_file.write(json.dumps(processed_data, ensure_ascii=False))
