import json

with open('result_model_we1_bert2.json','r', encoding='utf8') as f_load:
    load_lists = json.load(f_load)

    for i in range(0,len(load_lists)):
        for j in range(0,len(load_lists[1])):
            load_lists[i][j] = (load_lists[i][j][:-1]+'<|endoftext|>'+load_lists[i][j][-1:])

with open('result_model_we1_bert2_add.json', 'w', encoding='utf8') as f:
    json.dump(load_lists, f, indent=2, ensure_ascii=False)