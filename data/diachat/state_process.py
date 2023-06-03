
import json
import os
import copy
import jieba
from collections import defaultdict

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_source_name = "ontology.json"
goal_file_name = "../state.json"
data_source = os.path.join(current_dir, data_source_name)
goal_file = os.path.join(current_dir, goal_file_name)
state = {"基本信息": {}, "问题": {}, "饮食": {}, "运动": {}, "行为": {}, "治疗": {}}

with open(data_source, encoding="utf-8") as data_source:
    with open(goal_file, mode='w', encoding='utf-8') as goal_file:
        data_source = json.load(data_source)
        # goal_data = defaultdict(str)
        keyList = data_source.keys()
        for k in keyList:
            try:
                slotName, d_a = k.split("-")
                state[slotName][d_a] = ""
            except:
                print(k)
        # goal_data = goal_data.fromkeys(keyList, "")
        goal_file.write(json.dumps(state, ensure_ascii=False))
