"""
获取Infrom AskHow AskWhy AskFor AskForSure的domain slot 拼接为d-s
生成到infrom_ds.json askHow_ds.json askWhy_ds.json askfor_ds.json askforsure_ds.json
"""
import json

with open('convlab2/dpl/etc/data/source_data.json', 'r', encoding='UTF-8') as fp:
    full_data = json.load(fp)
    inform_file_data = []
    askhow_file_data = []
    askwhy_file_data = []
    askfor_file_data = []
    askforsure_file_data = []
    for session in full_data:
        for utterance in session["utterances"]:
            if utterance["agentRole"] == "User":
                for annotation in utterance["annotation"]:
                    act = annotation["act_label"]
                    if act in ["Inform", "AskHow", "AskWhy", "AskFor", "AskForSure"]:
                        for dsv in annotation["slot_values"]:
                            domain = dsv["domain"] if dsv["domain"] else "none"
                            slot = dsv["slot"] if dsv["domain"] else "none"
                            ds = '-'.join((domain, slot))
                            if act == "Inform" and ds not in inform_file_data:
                                inform_file_data.append(ds)
                            if act == "AskHow" and ds not in askhow_file_data:
                                askhow_file_data.append(ds)
                            if act == "AskWhy" and ds not in askwhy_file_data:
                                askwhy_file_data.append(ds)
                            if act == "AskFor" and ds not in askfor_file_data:
                                askfor_file_data.append(ds)
                            if act == "AskForSure" and ds not in askforsure_file_data:
                                askforsure_file_data.append(ds)
    with open('convlab2/dpl/etc/data/inform_ds.json', mode='w', encoding='UTF-8') as f:
        f.write(json.dumps(inform_file_data, ensure_ascii=False))
    with open('convlab2/dpl/etc/data/askhow_ds.json', mode='w', encoding='UTF-8') as f:
        f.write(json.dumps(askhow_file_data, ensure_ascii=False))
    with open('convlab2/dpl/etc/data/askwhy_ds.json', mode='w', encoding='UTF-8') as f:
        f.write(json.dumps(askwhy_file_data, ensure_ascii=False))
    with open('convlab2/dpl/etc/data/askfor_ds.json', mode='w', encoding='UTF-8') as f:
        f.write(json.dumps(askfor_file_data, ensure_ascii=False))
    with open('convlab2/dpl/etc/data/askforsure_ds.json', mode='w', encoding='UTF-8') as f:
        f.write(json.dumps(askforsure_file_data, ensure_ascii=False))