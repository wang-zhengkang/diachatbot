import json
import os
import copy


def getCategory(act_label):
    if act_label == "Inform":
        return "现状"
    if act_label in ["Advice", "AdviceNot"]:
        return "建议"
    else:
        return "解释"

# 由于需求不同，这个列表是错的，暂时不改了，判断的时候直接将列表写出来判断，最好不用以下两个列表
arg_actlabel = ['Inform', 'Advice', 'AdviceNot', 'Explanation']
noarg_actlabel = ['Assure', 'Deny', 'Accept', 'AskWhy', 'GeneralAdvice', 'GeneralExplanation', 'Chitchat']

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
data_source_name = "annotations_20220501_2.json"
goal_file_name = "seg_annotations.json"
data_source = os.path.join(current_dir, data_source_name)
goal_file = os.path.join(current_dir, goal_file_name)

with open(data_source, encoding="utf-8") as data_source:
    with open(goal_file, mode='w', encoding='utf-8') as goal_file:
        # 加载数据，进行备份
        data_source = json.load(data_source)
        # 目标数据
        goal_data = []
        # goal_file.write(json.dumps(data_source, ensure_ascii=False))

        # 遍历每一段对话
        for _, conversation in enumerate(data_source):
            # 初始化 user_state, sys_state, sys_state_init
            user_state = []
            sys_state_init = {}
            sys_state = {}

            # 存储上一个角色的annotation
            user_annotation = []
            sys_annotation = []

            # domains
            domains = []

            try:
                conversation = {
                    'conversationId': conversation['conversationId'],
                    'type': conversation['type'],
                    'domains': domains,
                    'agentId': conversation['agentId'],
                    'utterances': conversation['utterances']
                }
            except:
                conversation = {
                    'conversationId': conversation['conversationId'],
                    'type': conversation['type'],
                    'domains': domains,
                    'utterances': conversation['utterances']
                }
            # 遍历每句话
            for i, utterance in enumerate(conversation['utterances']):
                # sys的每一句话中sys_state_init, sys_state只更新一次
                sys_state_init_flag = True
                sys_state_flag = True
                for _, annotation in enumerate(utterance['annotation']):
                    # user_state: sys_annotation, user_annotation
                    if annotation['act_label'] in arg_actlabel:
                        for s in annotation['slot_values']:
                            user_state_temp = [s["domain"], s["slot"], s["value"]]
                            user_state.append(user_state_temp)
                    # sys_state_init: sys_state, user_annotation
                    if sys_state_init_flag:
                        sys_state_init = copy.deepcopy(sys_state)
                        sys_state_init_flag = False
                    if utterance['agentRole'] == 'User' and annotation['act_label'] in arg_actlabel:
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
                    if utterance['agentRole'] == 'Doctor' and annotation['act_label'] in arg_actlabel:
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
                                            sys_state_temp[s['domain']][category][num-1].update({'是否建议': 'true'})
                                        elif annotation['act_label'] == "AdviceNot":
                                            sys_state_temp[s['domain']][category][num-1].update({'是否建议': 'false'})
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
                    if annotation["act_label"] in ['Accept', 'Assure', 'Deny', 'AskWhy', 'Uncertain', 'AskForSure'] and len(utterance['annotation']) == 1:
                        for s in annotation["slot_values"]:
                            if s['domain'] not in domains and s['domain'] != '':
                                domains.append(s["domain"])
                        domain_temp += befor_domain_temp
                        domain_temp = list(set(domain_temp))
                    if annotation["act_label"] in ['GeneralAdvice', 'GeneralExplanation', 'Chitchat'] and len(utterance['annotation']) == 1:
                        continue
                for domain in domain_temp:
                    if conversation['utterances'][i]['domain'] == '':
                        conversation['utterances'][i]['domain'] = domain
                    else:
                        conversation['utterances'][i]['domain'] = conversation['utterances'][i][
                                                                  'domain'] + '-' + domain
                befor_domain_temp = domain_temp

            conversation['domains'] = domains
            goal_data.append(conversation)
        goal_file.write(json.dumps(goal_data, ensure_ascii=False))
