"""
获取AskFor AskForSure的domain slot 拼接为d-s
生成到askfor_ds.json and askforsure-ds.json
"""
import json
from pprint import pprint

with open('convlab2/policy/gdpl/diachat/data/source_data.json', 'r', encoding='UTF-8') as fp:
    full_data = json.load(fp)
    askfor_file_data = []
    askforsure_file_data = []
    for session in full_data:
        for utterance in session["utterances"]:
            if utterance["agentRole"] == "User":
                for annotation in utterance["annotation"]:
                    act = annotation["act_label"]
                    if act in ["AskFor", "AskForSure"]:
                        for dsv in annotation["slot_values"]:
                            domain = dsv["domain"]
                            slot = dsv["slot"]
                            ds = '-'.join((domain, slot))
                            if act == "AskFor" and ds not in askfor_file_data:
                                askfor_file_data.append(ds)
                            if act == "AskForSure" and ds not in askforsure_file_data:
                                askforsure_file_data.append(ds)
    with open('convlab2/policy/gdpl/diachat/data/askfor_ds.json', mode='w', encoding='UTF-8') as f:
        f.write(json.dumps(askfor_file_data, ensure_ascii=False))
    with open('convlab2/policy/gdpl/diachat/data/askforsure_ds.json', mode='w', encoding='UTF-8') as f:
        f.write(json.dumps(askforsure_file_data, ensure_ascii=False))