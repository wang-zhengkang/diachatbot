"""
文件预处理步骤：
1. 对源数据文件(annotations_20220627_2.json)进行add_nothanks_for_627_2.py生成annotations_nothanks.json
2. 对annotations_nothanks.json进行goal_preprocess.py生成annotations_goal.json
"""
import json
from pprint import pprint

with open("data/diachat/annotations_20220627_2.json", "r", encoding="utf-8") as fp:
    full_data = json.load(fp)
    new_file_data = []
    add_list = []
    utteranceId = 0
    for session in full_data:
        sequenceNo = 1
        conversationId = session["conversationId"]
        utterances = session["utterances"]
        utterances_num = len(utterances)
        utteranceId += utterances_num
        if utterances_num % 2 != 0:
            # add_list用于检查是否add成功
            add_list.append(conversationId)
            session["utterances"].append(
                {
                    "utteranceId": utteranceId,
                    "conversationId": conversationId,
                    "sequenceNo": utterances_num + 1,
                    "utterance": "不客气",
                    "agentRole": "Doctor",
                    "annotation": [
                        {
                            "act_label": "Chitchat",
                            "slot_values": [
                                {"domain": "", "slot": "", "value": "", "pos": ""}
                            ],
                        }
                    ],
                }
            )
        new_file_data.append(session)
    pprint(add_list)
    with open(
        "data/diachat/annotations_nothanks.json", mode="w", encoding="utf-8"
    ) as f:
        f.write(json.dumps(new_file_data, ensure_ascii=False))
