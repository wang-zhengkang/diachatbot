
import json
import os
import copy
import jieba
import collections



def sentseg(sent):
    sent = sent.replace('\t', ' ')
    sent = ' '.join(sent.split())
    tmp = " ".join(jieba.cut(sent))
    return ' '.join(tmp.split())


current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_source_name = "annotations.json"
goal_file_name = "ontology.json"
data_source = os.path.join(current_dir, data_source_name)
goal_file = os.path.join(current_dir, goal_file_name)

noargs_act = ['Accept', 'AskWhy', 'Assure', 'Deny', 'GeneralAdvice', 'Chitchat', 'GeneralExplanation']
dic=collections.defaultdict(int)
with open(data_source, encoding="utf-8") as data_source:
    with open(goal_file, mode='w', encoding='utf-8') as goal_file:
        data_source = json.load(data_source)
        goal_data = {}
        for _, conversation in enumerate(data_source):
            for _, utterance in enumerate(conversation['utterances']):
                for _, annotation in enumerate(utterance['annotation']):
                    if annotation['act_label'] in noargs_act:
                        continue
                    else:
                        act_temp = annotation['act_label']
                        if act_temp == "Inform":
                            act_temp = "现状"
                        elif act_temp in ["Advice", "AdviceNot"]:
                            act_temp = "建议"
                        elif act_temp =="Explanation":
                            act_temp = "解释"
                        else:
                                dic[act_temp]+=1
                                continue
                        for s in annotation['slot_values']:
                            # key = s['domain'] + '-' + s['slot'] + '-' + act_temp 
                            key = s['domain'] + '-' + s['slot']
                            if key not in goal_data:
                                print(key)
                                goal_data[key] = []
                            if s['value'] in goal_data[key] or s['value'] == '':
                                continue
                            goal_data[key].append(sentseg(s['value']))
        goal_file.write(json.dumps(goal_data, ensure_ascii=False))
        print(dic)
