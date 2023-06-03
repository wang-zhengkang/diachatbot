import json
with open('results2.json','r', encoding='utf8') as f_load:
    result_model_we1_bert1 = json.load(f_load)
with open('final_test2.json','r', encoding='utf8') as f_load:
    test = json.load(f_load)
with open('final_result2.json', 'w', encoding='utf8') as f:
    for i in range(0,len(result_model_we1_bert1)):
        test[i]=test[i][:2]+result_model_we1_bert1[i]
    json.dump(test, f, indent=2, ensure_ascii=False)