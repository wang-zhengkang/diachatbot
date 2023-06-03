"""
generate sys_da_voc.txt & usr_da_voc.txt
"""
import os
import json
from pprint import pprint

current_dir = os.path.dirname(os.path.abspath(__file__))  # ..\diachatbot\data\diachat

data_source_name = "annotations_goal.json"
sys_da_voc_name = "sys_da_voc.txt"
usr_da_voc_name = "usr_da_voc.txt"

data_source_dir = os.path.join(current_dir, data_source_name)
sys_da_voc_dir = os.path.join(current_dir, sys_da_voc_name)
usr_da_voc_dir = os.path.join(current_dir, usr_da_voc_name)


def value2num(source_list: list, target_list: list):
    count = 1
    judge_list = ["*"]
    for item in source_list:
        temp_list = item.split('-')
        flag = set(judge_list) <= set(temp_list)
        count = count + 1 if flag else 1
        judge_list = [temp_list[0], temp_list[1], temp_list[2]]
        temp_list[3] = count
        target_list.append(temp_list[0] + '-' + temp_list[1] + '-' + temp_list[2] + '-' + str(temp_list[3]))


def writefile(source_list, target_dir: str):
    with open(target_dir, 'w', encoding="utf-8") as f:
        for item in source_list:
            f.write(item)
            f.write('\n')


if __name__ == '__main__':
    if not os.path.exists(sys_da_voc_dir):
        file = open(sys_da_voc_dir, 'w')
        file.close()
    if not os.path.exists(usr_da_voc_dir):
        file = open(usr_da_voc_dir, 'w')
        file.close()

    with open(data_source_dir, encoding="utf-8") as data_source:
        data_source = json.load(data_source)
        sys_da_voc_list = []
        usr_da_voc_list = []
        for _, conversation in enumerate(data_source):
            for _, utterance in enumerate(conversation['utterances']):
                role = utterance['agentRole']
                for _, annotation in enumerate(utterance['annotation']):
                    act_label = annotation['act_label']
                    for dsv in annotation['slot_values']:
                        domain = dsv['domain'] if dsv['domain'] != "" else 'none'
                        slot = dsv['slot'] if dsv['slot'] != "" else 'none'
                        value = dsv['value'] if dsv['value'] != "" else 'none'
                        adsv = act_label + '-' + domain + '-' + slot + '-' + value
                        if role == 'User':
                            if adsv not in usr_da_voc_list:
                                usr_da_voc_list.append(adsv)
                        else:
                            if adsv not in sys_da_voc_list:
                                sys_da_voc_list.append(adsv)
        usr_da_voc_list.sort()
        sys_da_voc_list.sort()
        usr_da_voc_devalue_list = []
        sys_da_voc_devalue_list = []
        value2num(usr_da_voc_list, usr_da_voc_devalue_list)
        value2num(sys_da_voc_list, sys_da_voc_devalue_list)
        writefile(usr_da_voc_devalue_list, usr_da_voc_dir)
        writefile(sys_da_voc_devalue_list, sys_da_voc_dir)