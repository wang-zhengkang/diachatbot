#encoding=UTF-8
'''
Created on 2022年4月15日

@author: yangjinfeng
'''

'''
+─────────────────────────+────────────────────+─────────────────────+──────────────────────────────+
| three major categories  | recommand_type     | recommend_method    | description                  |
+─────────────────────────+────────────────────+─────────────────────+──────────────────────────────+
|                         | Advice             | advice              |advice sth or advice not sth  |
| three specific answers  | Assure             | assure              |assure sth or deny sth        |
|_________________________| Explanation        | explanation         |explanation by sth            |
|                         | AskFor             | ask_for             |ask for sth                   |
| three specific questions| AskForSure         | ask_for_sure        |to make sth sure              |
|_________________________| AskHow             | ask_how             |ask about how sth is going    |
|                         | GeneralAdvice      | general_advice      |give some general advice      |
| three general responses | GeneralExplanation | general_explanation |give some general explanation |
|                         | Chitchat           | chitchat            |including greetings, thanks   |    
+─────────────────────────+────────────────────+─────────────────────+──────────────────────────────+
    
'''
import csv
import os
import re
import json
from diachatbot.ai.infer import Inference
import jieba
from diachatbot.ai.ckan.CKAN_robot_interface import predict
from diachatbot.ai.ckan.CKAN_tokenization import user_id_sym_dis_id

class SimpleInference(Inference):
    
    def __init__(self):
        db_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_list = list(csv.reader(open(os.path.join(db_dir,"database.csv"), encoding='utf-8')))


    def advice(self,state,domain,slots=None):
        '''
                       建议什么或者不建议什么
                        返回 [{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n, flag:False}]
            flag:False表示不建议
        '''
        #print(state,domain,slots)
        if domain == '饮食':
            user_id_sym_dis_id(state)#根据不同的state生成不同的predict_patient_disease_symptom.npy文件
            recommend_food_id, recommend_food_name, recommend_probability=predict()#加载文件，预测当前state下的回答
            #res = {}
            #res[slots[0]] = recommend_food_name[0]+'、'+recommend_food_name[1]+'、'+recommend_food_name[2]
            res = []
            for food_name in recommend_food_name[0:3]:#取前三个回答，回答太多NLG无法生成答案
                res.append({slots[0]:food_name})
            return res

        if state["askfor_slots"] == []:
            AskFor_slot_value = str(state["belief_state"])
        if state["askfor_slots"] != []:
            AskFor_slot_value =state["askfor_slots"][0][1] +  state["askfor_slots"][0][2]

        data_list = self.data_list
        ress_total = []
        if len(state["belief_state"]) == 0:
            res = []
            for slot in slots:
                res.append({slot:'不知道'})
            return res
        feature = list(state["belief_state"].values())[0]
        feature = list(feature.values())[0]
        if len(feature) > 2:
            feature0 = feature[0]
            feature1 = feature[1]
            feature2 = feature[2]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc.append(self.removePunctuation(str(feature1)))
            all_doc.append(self.removePunctuation(str(feature2)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.5:
            #for data in data_list:
                #if str(feature0) in str(data) and str(feature1) in str(data) and str(feature2) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '建议':
                                    for slot in slots:
                                        true_change = re.sub(r'True', '\'True\'', str(data1[3]))
                                        Dicts = json.loads(re.sub(r'\'', '"', true_change))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        sim_advice = len(set(str(Dict)) & set(AskFor_slot_value)) / len(set(str(Dict)) | set(AskFor_slot_value))
                                                        if sim_advice >= 0.22:
                                                            ress_total.append(Dict)
        if len(feature) == 2:
            feature0 = feature[0]
            feature1 = feature[1]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc.append(self.removePunctuation(str(feature1)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.4:
            #for data in data_list:
                #if str(feature0) in str(data) and str(feature1) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '建议':
                                    for slot in slots:
                                        true_change = re.sub(r'True', '\'True\'', str(data1[3]))
                                        Dicts = json.loads(re.sub(r'\'', '"', true_change))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        sim_advice = len(set(str(Dict)) & set(AskFor_slot_value)) / len(set(str(Dict)) | set(AskFor_slot_value))
                                                        if sim_advice >= 0.22:
                                                            ress_total.append(Dict)
        if len(feature) == 1:
            feature0 = feature[0]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.25:
            #for data in data_list:
                #if str(feature0) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '建议':
                                    for slot in slots:
                                        true_change = re.sub(r'True', '\'True\'', str(data1[3]))
                                        Dicts = json.loads(re.sub(r'\'', '"', true_change))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        sim_advice = len(set(str(Dict)) & set(AskFor_slot_value)) / len(set(str(Dict)) | set(AskFor_slot_value))
                                                        if sim_advice >= 0.22:
                                                            ress_total.append(Dict)
        res = []
        [res.append(x) for x in ress_total if x not in res]
        if res ==[]:
            for slot in slots:
                res.append({slot:'不知道'})
        #for res_piece in res:
            #if 'True' in str(res_piece):
                #res.remove(res_piece)
        return res



    def assure(self,state,domain,slots=None):
        '''
                    用户的act是AskForSure的时候,系统给出肯定或者否定，用户 AskForSure以及slot-value)记录在state里
                    返回用户AskForSure携带的slot-value，形如：
        [{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n, flag:False}]
        flag:False表示Deny
        '''
        if state["askforsure_slotvs"] != []:
            AskForSure_value = state["askforsure_slotvs"][0][2]
        else:
            AskForSure_value = ''
        data_list = self.data_list
        ress_total = []
        if len(state["belief_state"]) == 0:
            res = []
            for slot in slots:
                res.append({slot:''})
            return res
        feature = list(state["belief_state"].values())[0]
        feature = list(feature.values())[0]
        if len(feature) > 2:
            feature0 = feature[0]
            feature1 = feature[1]
            feature2 = feature[2]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc.append(self.removePunctuation(str(feature1)))
            all_doc.append(self.removePunctuation(str(feature2)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                #if sim >= 0.5:
                if AskForSure_value in doc_test:
            #for data in data_list:
                #if str(feature0) in str(data) and str(feature1) in str(data) and str(feature2) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '疑问':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 ="'"+slot+'\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        res = []
        [res.append(x) for x in ress_total if x not in res]
        if res == []:
            for slot in slots:
                res.append({slot: ''})
        return res

    def explanation(self,state,domain,slots=None):
        '''
                        希望给出具体的解释，返回值体现了解释的信息要素
                        返回 [{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n}]
                        state = {"user_action": [{"act_label": "AskForSure", "slot_values": [{"domain": "行为", "slot": "行为名", "value": "少喝", "pos": "2-4"}]}],
                        "belief_state": {'饮食': {'现状': [{'饮食名': '莲藕排骨'}, {'饮食名': '汤'}]}, '问题': {'现状': [{'血糖值': '升高'}]}},
                        "askfor_slots": [[]],
                        "askforsure_slotvs": [['行为', '行为名', '少喝']]}
                        delex_das = [['Explanation', '饮食', '饮食名'], ['Explanation', '饮食', '饮食量']]
        '''
        data_list = self.data_list
        ress_total = []
        if len(state["belief_state"]) == 0:
            res = []
            for slot in slots:
                res.append({slot:'不知道'})
            return res
        feature = list(state["belief_state"].values())[0]
        feature = list(feature.values())[0]
        if len(feature) > 2:
            feature0 = feature[0]
            feature1 = feature[1]
            feature2 = feature[2]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc.append(self.removePunctuation(str(feature1)))
            all_doc.append(self.removePunctuation(str(feature2)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.5:
            #for data in data_list:
                #if str(feature0) in str(data) and str(feature1) in str(data) and str(feature2) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '解释':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 ="'"+slot+'\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        if len(feature) == 2:
            feature0 = feature[0]
            feature1 = feature[1]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc.append(self.removePunctuation(str(feature1)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.4:
            #for data in data_list:
                #if str(feature0) in str(data) and str(feature1) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '解释':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 ="'"+slot+'\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        if len(feature) == 1:
            feature0 = feature[0]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.25:
            #for data in data_list:
                #if str(feature0) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '解释':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 = "'" + slot + '\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        res =[]
        [res.append(x) for x in ress_total if x not in res]
        if res ==[]:
            for slot in slots:
                res.append({slot:'不知道'})
        return res



    def ask_for(self,state,domain,slots=None):
        '''
                    系统给出要追问的问题，就是推断出追问的槽位
                    返回[{slot1:value1,slot2:},{slot_m:value_m,slot_n:}]

        '''
        data_list = self.data_list
        ress_total = []
        if len(state["belief_state"]) == 0:
            res = []
            for slot in slots:
                res.append({slot:'不知道'})
            return res
        feature = list(state["belief_state"].values())[0]
        feature = list(feature.values())[0]
        if len(feature) > 2:
            feature0 = feature[0]
            feature1 = feature[1]
            feature2 = feature[2]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc.append(self.removePunctuation(str(feature1)))
            all_doc.append(self.removePunctuation(str(feature2)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.5:
                #if str(feature0) in str(data) and str(feature1) in str(data):
            #for data in data_list:
                #if str(feature0) in str(data) and str(feature1) in str(data) and str(feature2) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '疑问':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 ="'"+slot+'\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        if len(feature) == 2:
            feature0 = feature[0]
            feature1 = feature[1]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc.append(self.removePunctuation(str(feature1)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.4:
                #if str(feature0) in str(data) and str(feature1) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '疑问':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 ="'"+slot+'\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        if len(feature) == 1:
            feature0 = feature[0]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.25:
            #for data in data_list:
                #if str(feature0) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '疑问':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 = "'" + slot + '\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        res =[]
        [res.append(x) for x in ress_total if x not in res]
        if res ==[]:
            for slot in slots:
                res.append({slot:'不知道'})
        return res


    def ask_for_sure(self,state,domain,slots=None):
        '''
                    系统推断出患者可能的现状，并希望用户予以确认
                    返回[{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n}]
        '''
        data_list = self.data_list
        ress_total = []
        if len(state["belief_state"]) == 0:
            res = []
            for slot in slots:
                res.append({slot:'不知道'})
            return res
        feature = list(state["belief_state"].values())[0]
        feature = list(feature.values())[0]
        if len(feature) > 2:
            feature0 = feature[0]
            feature1 = feature[1]
            feature2 = feature[2]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc.append(self.removePunctuation(str(feature1)))
            all_doc.append(self.removePunctuation(str(feature2)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.5:
            #for data in data_list:
                #if str(feature0) in str(data) and str(feature1) in str(data) and str(feature2) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '疑问':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 ="'"+slot+'\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        if len(feature) == 2:
            feature0 = feature[0]
            feature1 = feature[1]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc.append(self.removePunctuation(str(feature1)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.4:
            #for data in data_list:
                #if str(feature0) in str(data) and str(feature1) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '疑问':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 ="'"+slot+'\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        if len(feature) == 1:
            feature0 = feature[0]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.25:
            #for data in data_list:
                #if str(feature0) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '疑问':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 = "'" + slot + '\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        res =[]
        [res.append(x) for x in ress_total if x not in res]
        if res ==[]:
            for slot in slots:
                res.append({slot:'不知道'})
        return res



    def ask_how(self,state,domain,slots=None):
        '''
                    表示医生问患者的行为、饮食等情况，arguments表示与问题相关的信息，
        slot是有value的，表示具体的行为、治疗、饮食是如何实施的。
                    这类问题可以看成是笼统的ask_for，在口语对话中经常会出现ask_how这样的问句，比如“你晚饭咋吃的呢”，这个问题可以包含“吃了什么”、“吃了多少”、“啥时候吃的”。
                    返回[{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n}]
        '''
        data_list = self.data_list
        ress_total = []
        if len(state["belief_state"]) == 0:
            res = []
            for slot in slots:
                res.append({slot:'不知道'})
            return res
        feature = list(state["belief_state"].values())[0]
        feature = list(feature.values())[0]
        if len(feature) > 2:
            feature0 = feature[0]
            feature1 = feature[1]
            feature2 = feature[2]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc.append(self.removePunctuation(str(feature1)))
            all_doc.append(self.removePunctuation(str(feature2)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.5:
            #for data in data_list:
                #if str(feature0) in str(data) and str(feature1) in str(data) and str(feature2) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '疑问':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 ="'"+slot+'\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        if len(feature) == 2:
            feature0 = feature[0]
            feature1 = feature[1]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc.append(self.removePunctuation(str(feature1)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.4:
            #for data in data_list:
                #if str(feature0) in str(data) and str(feature1) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '疑问':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 ="'"+slot+'\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        if len(feature) == 1:
            feature0 = feature[0]
            all_doc = []
            all_doc.append(self.removePunctuation(str(feature0)))
            all_doc_list = []
            for doc in all_doc:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
            all_doc = ''
            for doc_list in all_doc_list:
                for word in doc_list:
                    all_doc = all_doc + word
            for data in data_list:
                doc_test_list = list(jieba.cut(self.removePunctuation(str(data))))
                doc_test = ''
                for word in doc_test_list:
                    doc_test = doc_test + word
                sim = len(set(all_doc) & set(doc_test)) / len(set(all_doc) | set(doc_test))
                if sim >= 0.25:
            #for data in data_list:
                #if str(feature0) in str(data):
                    for data1 in data_list:
                        if data1[0] == data[0]:
                            if data1[1] == domain:
                                if data1[2] == '疑问':
                                    for slot in slots:
                                        string = data1[3]
                                        Matching1 = "'" + slot + '\': \'.*\''
                                        Dicts = json.loads(re.sub(r'\'', '"', str(data1[3])))
                                        for Dict in Dicts:
                                            for key in list(Dict.keys()):
                                                if slot != key:
                                                    del Dict[key]
                                                if slot == key:
                                                    Dict_value = {}
                                                    Dict_value[key] = Dict[key]
                                                    set_value = 0
                                                    for slot_ in slots:
                                                        if slot_ not in str(Dict):
                                                            set_value = 1
                                                    if set_value == 0:
                                                        ress_total.append(Dict)
        res =[]
        [res.append(x) for x in ress_total if x not in res]
        if res ==[]:
            for slot in slots:
                res.append({slot:'不知道'})
        return res


    '''
                系统给出泛泛的回应，不考虑用户的当前情况。   z这三类推荐可什么都不做，下一步直接交给NLG部分生成回复。
    '''
    def general_advice(self,state,domain):

        return []


    def general_explanation(self,state,domain):

        return []

    def chitchat(self,state):

        return []
    
    
    def removePunctuation(self, content):
        """
                        文本去标点
        """
        punctuation = r"~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./，。、《》 ？；：‘“{【】}|、！@#￥%……&*（）——+=-"
        content = re.sub(r'[{}]+'.format(punctuation), '', content)
        return content.strip().lower()

if __name__ == '__main__':
    #path = '输入.json'
    fold_path = 'test_data'
    path = 'input'
    for num in range(1,5):
        f = open(fold_path + '//' + path + str(num) + '.json', 'r', encoding='utf-8')
        policy_output = json.load(f)
        inf = Inference()
        for num in range(0,1):
            str_num = str(num)
            print("prediction：",policy_output[str_num]['prediction'])
            kg_action = inf.recommend_by_delex_da(policy_output[str_num]['state'], policy_output[str_num]['prediction'])
        #print("golden_action:",policy_output[str_num]['golden_action'])#这是希望得到的输出
            print("kg_action:", kg_action)#这是实际输出