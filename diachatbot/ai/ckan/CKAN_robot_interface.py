# -*- coding: utf-8 -*-
# @Author  : Haochen
# @Time    : 2022/6/27 15:48
# @Function:

"""
This code is used to interface with robbot and Prediciton code
"""

import joblib
import collections
import os
import numpy as np
import argparse
import torch
import numpy as np
from diachatbot.ai.ckan.CKAN_data_loader import load_data
from diachatbot.ai.ckan.CKAN_train import train , ctr_eval, topk_eval
#from diachatbot.ai.CKAN.CKAN_model import CKAN
#from CKAN_model import CKAN
import warnings
warnings.filterwarnings('ignore')

#logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='medicine', help='which dataset to use')
parser.add_argument('--n_epoch', type=int, default=1, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--kg_batch_size', type=int, default=5000, help='kg batch size')
parser.add_argument('--n_layer', type=int, default=3, help='depth of layer')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--dim', type=int, default=32, help='dimension of entity and relation embeddings')
parser.add_argument('--user_triple_set_size', type=int, default=128, help='the number of triples in triple set of user')
parser.add_argument('--item_triple_set_size', type=int, default=32, help='the number of triples in triple set of item')
parser.add_argument('--agg', type=str, default='concat', help='the type of aggregator (sum, pool, concat)')
parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5)
parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5)
parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

def sort_n(y_score_list,x=5):
    sort_y_sore = np.sort(y_score_list)
    # print("sort_a,从小到大的index:", sort_y_sore)
    T_a = sort_y_sore[::-1]
    # print("最大的5位的index:", T_a[:x])
    return T_a[:x]
def find_index(max_n,y_score_list):
    recommend_food_index1 = []
    for i in max_n:
        index_x = y_score_list.index(i)
        recommend_food_index1.append(index_x)
    # print('index:', recommend_food_index1)
    return recommend_food_index1
def find_id(recommend_food_index1,y_food_entity_id):
    recommend_food_id1 = []
    for i in recommend_food_index1:
        recommend_food_id_x = y_food_entity_id[i]
        recommend_food_id1.append(recommend_food_id_x)
    # print(recommend_food_id1)
    return recommend_food_id1
def find_rec_food(recommend_food_id,dictTmp):
    recommend_food_name1 = []
    for i in recommend_food_id:
        recommend_food_name1.append(dictTmp[i])
    return recommend_food_name1

def predict():
    args = parser.parse_args()
    data_info = load_data(args)
    file_path = os.path.dirname(__file__)
    model = joblib.load(file_path+'//src//CKAN_model.pkl')
    #model = joblib.load('CKAN_model.pkl')
    predict_data = data_info[0]
    user_triple_set = data_info[3]
    item_triple_set = data_info[4]

    #logging.info('输出推荐结果......')
    y_patient_id,y_food_entity_id,y_true_list,y_score_list,y_pred_list = ctr_eval(args, model, predict_data, user_triple_set, item_triple_set)
    # ctr_info = 'epoch %.2d    test auc: %.4f f1: %.4f pre: %.4f recall: %.4f'

    # max_probability=max(y_score_list)
    # recommend_food_index=y_score_list.index(max_probability)
    # recommend_food_id=y_food_entity_id[recommend_food_index]
    # print('推荐食物ID：',recommend_food_id,'   推荐概率为：',max_probability)

    ##默认最大五个
    max_five = sort_n(y_score_list)
    ##find index
    recommend_food_index1 = find_index(max_five,y_score_list)
    ##find id
    recommend_food_id1 = find_id(recommend_food_index1,y_food_entity_id)

    #print('推荐食物ID：', recommend_food_id1, '   推荐概率为：', max_five)

    entity_dict_file = '\\data\\entity_dict.txt'
    file_path = os.path.dirname(__file__)
    f = open(file_path + entity_dict_file, encoding="utf-8", )
    entity_dict_str = f.read()
    entity_dict = eval(entity_dict_str)
    # print(entity_dict)
    dictTmp_1 = {}
    for entity, ent_id in entity_dict.items():
        dictTmp_1[ent_id] = entity

    # recommend_food_name = dictTmp_1[recommend_food_id]
    recommend_food_name1 = find_rec_food(recommend_food_id1, dictTmp_1)
    # print('推荐食物名称：',recommend_food_name1)
    # print('根据您的疾病症状特征，建议您吃适量的：', recommend_food_name1)
    # print('根据您的疾病症状特征，建议您吃适量的：',recommend_food_name)
    return recommend_food_id1,recommend_food_name1,max_five


if __name__ =='__main__':
    predict()
























