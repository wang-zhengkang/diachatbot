import json
import zipfile
from collections import Counter
from pprint import pprint
from convlab2.dst.rule.diachat.dst import RuleDST
from convlab2.dst.rule.diachat.util.state_structure import default_state
from convlab2.util.diachat.state_structure import validate_domain_group
from copy import deepcopy


def calculateJointState(predict_golden):
    res = []
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        res.append(predicts==labels)
    return sum(res) / len(res) if len(res) else 0.


def calculateSlotState(predict_golden):
    res = []
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        for x, y in zip(predicts, labels):
            for w, z in zip(predicts[x].values(),labels[y].values()):
                res.append(w==z)
    return sum(res) / len(res) if len(res) else 0.


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))
# 将 sys_state 或者 sys_state_init 添加到 belief_state 中
def sys_belief_state(sys_state_dict,ruleDST):
    for domain, svs in sys_state_dict.items():
        for label, slotValue in svs.items():
            if not validate_domain_group(domain,label):
                continue
            for j, slotvalue in enumerate(slotValue):
                for slot, value in slotvalue.items():
                    try:
                        if slot != "" and (slot not in['是否建议','状态']):
                            num = len(ruleDST.state['belief_state'][domain][label])
                            if not ruleDST.state['belief_state'][domain][label][num - 1][slot]:
                                ruleDST.state['belief_state'][domain][label][num - 1][slot] = value
                            else:
                                new_entry = default_state()['belief_state'][domain][label][0]
                                new_entry[slot] = value
                                ruleDST.state['belief_state'][domain][label].append(new_entry)
                    except Exception as e :
                        print(e)

def evaluate_sys_state(data):
    ruleDST = RuleDST()
    ruleDST_init = RuleDST()
    state_predict_golden = []
    for task_id, item in enumerate(data):
        for i, turn in enumerate(item['utterances']):
            if turn['agentRole'] == 'Doctor':
                ruleDST.init_session()
                ruleDST_init.init_session() #每一轮对话里的那个sys_state_init，就是截止到当前的跟踪结果
                usr_da = []
                # 上一轮用户说的话：提取 intent,domain,slot,value 保存在 usr_da 中 ，并添加到 belief_state 中
                for j, dialogue_act in enumerate(item['utterances'][i - 1]['annotation']):
                    for k,slot_value in enumerate(dialogue_act["slot_values"]):
                        dialog_act = []
                        dialog_act.append(dialogue_act['act_label'])
                        dialog_act.append(slot_value["domain"])
                        dialog_act.append(slot_value["slot"])
                        dialog_act.append(slot_value["value"])
                        usr_da.append(dialog_act)
                ruleDST.update(usr_da)
                # 上一轮系统说的话：将 sys_state 添加到 ruleDST 的 belief_state 中
                if i > 2:
#                     print('utteranceId:',item['utterances'][i - 2]['utteranceId'])
#                     print('conversationId:',item['utterances'][i - 2]['conversationId'])
#                     print('sequenceNo',item['utterances'][i - 2]['sequenceNo'])
                    
                    sys_belief_state(item['utterances'][i - 2]['sys_state'],ruleDST)
                new_state = deepcopy(ruleDST.state['belief_state'])
                # 本轮的 sys_state_init 将 sys_state_init 添加到 ruleDST_init 的 belief_state 中
                sys_belief_state(turn['sys_state_init'],ruleDST_init)
                golden_state = deepcopy(ruleDST_init.state['belief_state'])
                state_predict_golden.append({
                        'predict': new_state,
                        'golden': golden_state
                })
    print('joint state', calculateJointState(state_predict_golden))
    print('slot state', calculateSlotState(state_predict_golden))


if __name__ == '__main__':
    data_path = "dst_evaluate_data0.json"
    data = json.load(open(data_path,encoding='utf-8'))
    evaluate_sys_state(data)
    
    data_path = "ruledst_evaluate_data.json"
    data = json.load(open(data_path,encoding='utf-8'))
    evaluate_sys_state(data)
