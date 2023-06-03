'''
1.每次都是医生先说，（医生第一句为空），修正有时以用户结尾时，医生最后一句话被改为“不客气”

2.修正之前belief_state很多为空的情况

3.简化问题，每回合相同的domain-slot只保留一个domain-slot-value，多回合逐步更新同一个domain-slot的value

'''
import json
import os
import jieba
import copy
from collections import defaultdict


def sentseg(sent):
    sent = sent.replace('\t', ' ')
    sent = ' '.join(sent.split())
    tmp = " ".join(jieba.cut(sent))
    return ' '.join(tmp.split())


def getCategory(act_label):
    '''
    本次dial数据未使用
    '''
    if act_label == "Inform":
        return "现状"
    if act_label in ["Advice", "AdviceNot"]:
        return "建议"
    else:
        return "解释"


current_dir = os.path.dirname(os.path.abspath(__file__))
data_source_name = "seg_annotations.json"
goal_file_name = "dial_annotations.json"
data_source = os.path.join(current_dir, data_source_name)
goal_file = os.path.join(current_dir, goal_file_name)

with open(data_source, encoding="utf-8") as data_source:
    with open(goal_file, mode='w', encoding='utf-8') as goal_file:
        data_source = json.load(data_source)
        data = []
        for i, conversation in enumerate(data_source):
            conversation_dic = {
                "dialogue_idx": str(conversation["conversationId"])
            }
            dialogue = []
            belief_state = []
            ds2id={}
            domains = []
            conversation_dic['domains'] = domains
            for j, utterance in enumerate(conversation["utterances"]):
                if j == 0 or j % 2 == 1:
                    sentence_dic = {
                        "system_transcript": "",
                        "transcript": "",
                        "turn_idx": 0,
                        "belief_state": copy.deepcopy(belief_state),
                        "domain": ""
                    }
                    domain_temp = []
                    domain_dic=defaultdict(int)

                if utterance["agentRole"] == "Doctor":
                    sentence_dic["system_transcript"] = sentseg(utterance["utterance"])
                if utterance["agentRole"] == "User":
                    sentence_dic["transcript"] = sentseg(utterance["utterance"])
                if utterance["sequenceNo"] == len(conversation["utterances"]):
                    # if utterance["agentRole"] == "Doctor":
                    sentence_dic["transcript"] = "谢谢您"
                    # if utterance["agentRole"] == "User":
                    #     sentence_dic["system_transcript"] = "不客气"

                # append belief_state

                for k, annotation in enumerate(utterance["annotation"]):


                    if annotation['act_label'] in ['Inform', 'Advice', 'AdviceNot', 'Explanation']:
                        # category = getCategory(annotation['act_label'])
                        pass
                    elif  utterance["sequenceNo"]==1:
                        pass
                    else:
                        continue

                    for s in annotation["slot_values"]:
                        belief_state_temp = {"slots": []}
                        temp = []
                        # temp_anno = s["domain"] + "-" + s["slot"] + "-" + category
                        temp_anno = s["domain"] + "-" + s["slot"]
                        domain_dic[temp_anno]+=1
                        if domain_dic[temp_anno]>1:   #以回合第一次读到的domain-slot-act为准，后续可考虑以最后一次读到的为准
                            continue

                        temp_value = sentseg(s["value"])
                        temp.append(temp_anno)
                        temp.append(temp_value)
                        # temp.append(category)
                        belief_state_temp["slots"].append(temp)
                        if temp_anno not in ds2id:
                            ds2id[temp_anno]=len(ds2id)
                            belief_state.append(belief_state_temp)
                        else:
                            belief_state[ds2id[temp_anno]]["slots"][0][1]=temp_value
                        sentence_dic["belief_state"] = copy.deepcopy(belief_state)

                        if s['domain'] not in domains:
                            domains.append(s["domain"])
                        if not domain_temp:
                            domain_temp.append(s["domain"])
                            sentence_dic['domain'] = s["domain"]
                        elif s['domain'] not in domain_temp:
                            domain_temp.append(s["domain"])
                            sentence_dic['domain'] = sentence_dic['domain'] + '-' + s["domain"]

                if j % 2 == 0 or utterance["sequenceNo"] == len(conversation["utterances"]):
                    dialogue.append(sentence_dic)

            conversation_dic["dialogue"] = dialogue
            conversation_dic["domains"] = domains

            data.append(conversation_dic)
        goal_file.write(json.dumps(data, ensure_ascii=False))
