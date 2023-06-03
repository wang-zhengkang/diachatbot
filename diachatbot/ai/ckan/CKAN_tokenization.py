import pandas as pd
import numpy as np
import json
import uuid
import sys
import os

def user_id_sym_dis_id(state):
    file_path = os.path.dirname(__file__)
    sym_dis_id=pd.read_csv(file_path+'//data//疾病症状ID.csv',header=0)
    sym_dis_id=sym_dis_id.drop_duplicates().reset_index()[['sym_dis','sym_dis_id']]
    sym_dis_list=[]
    for i in state['belief_state']:
        if i=='问题':
            for j in state['belief_state']['问题']['现状']:
                for k in j:
                    if k=='症状':
                        sym_dis_list.append(j[k])
                    elif k =='疾病':
                        sym_dis_list.append(j[k])
    #user_id=str(uuid.uuid1())
    user_id = 10001
    user_id_list=[user_id]*len(sym_dis_list)
    user_sym_dis=pd.DataFrame({'user_id':user_id_list,'sym_dis':sym_dis_list})
    dff=pd.merge(user_sym_dis,sym_dis_id,how='left',on='sym_dis')[['user_id','sym_dis_id']]
    dff['sign'] = dff.apply(lambda x: 1, axis=1)
    dff=dff.values
    np.save(file_path+'//src//predict_patient_disease_symptom.npy',dff)

def user_id_food_id():
    file_path = os.path.dirname(__file__)
    food_data = pd.read_csv(file_path+'//data//饮食ID.csv', header=0)
    food_id_list = []
    for index, row in food_data.iterrows():
        food_id_list.append(row['food_id'])
    #user_id = str(uuid.uuid1())
    user_id = 10001
    user_id_list = [user_id] * len(food_id_list)
    sign_list = [-1] * len(food_id_list)
    df = pd.DataFrame({'user_id': user_id_list, 'food_id': food_id_list, "sign": sign_list})
    df = df.values
    np.save('patient_all_food_id_for_predict_test.npy',df)

#txt文件会先清除再写入，如果不存在就会先创建再写入