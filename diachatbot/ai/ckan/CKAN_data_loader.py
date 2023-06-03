import collections
import os
import numpy as np
import logging
from functools import reduce

#logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    #logging.info("================== preparing data ===================")
    predict_data, user_init_entity_set, item_init_entity_set = load_rating(args)
    n_entity, n_relation, kg, kg_np = load_kg(args) 
    #logging.info("contructing users' kg triple sets ...")
    user_triple_sets = kg_propagation(args, kg, user_init_entity_set, args.user_triple_set_size, True)
    #logging.info("contructing items' kg triple sets ...")
    item_triple_sets = kg_propagation(args, kg, item_init_entity_set, args.item_triple_set_size, False)
    return predict_data, n_entity, n_relation, user_triple_sets, item_triple_sets, kg, kg_np


def load_rating(args):
    # existing_patient_food_file = 'patient_food'
    predict_patient_food_file = 'patient_all_food_id_for_predict_test'
    #logging.info("load rating file: %s.npy", predict_patient_food_file)
    ####所有原始的数据当作预测的训练集（但是不进行训练，只是为了尽可能少的修改代码）
    # if os.path.exists(existing_patient_food_file + '.npy'):
    #     train_rating_np = np.load(existing_patient_food_file + '.npy')
    # else:
    #     train_rating_np = np.loadtxt(existing_patient_food_file + '.txt', dtype=np.int32)
    #     np.save(existing_patient_food_file + '.npy', train_rating_np)
    ####对于要预测的患者，给他新生成他饮食的文档，id设置为新的未出现过的ID，food的id和是否吃食物都设置为-1
    file_path = os.path.dirname(__file__)
    if os.path.exists(file_path+'//src//'+predict_patient_food_file + '.npy'):
        predict_rating_np = np.load(file_path+'//src//'+predict_patient_food_file + '.npy', allow_pickle=True)
    else:
        predict_rating_np = np.loadtxt(file_path+'//src//'+predict_patient_food_file + '.txt', dtype=np.int32)
        np.save(predict_patient_food_file + '.npy', predict_rating_np)

    return dataset_split(predict_rating_np)


def dataset_split(predict_rating_np):
    #logging.info("preparing predict dataset  ......")

    user_init_entity_set, item_init_entity_set = collaboration_propagation(predict_rating_np)

    # train_data = train_rating_np
    predict_data = predict_rating_np
    
    return predict_data, user_init_entity_set, item_init_entity_set
    
    
def collaboration_propagation(predict_rating_np):
    entity_file = 'predict_patient_disease_symptom'
    # predict_file= 'predict_patient_disease_symptom'

    #logging.info("load predict rating file: %s.npy", entity_file)

    file_path = os.path.dirname(__file__)
    if os.path.exists(file_path+'//src//'+entity_file + '.npy'):
        entity_np = np.load(file_path+'//src//'+entity_file + '.npy',allow_pickle=True)
    else:
        entity_np = np.loadtxt(file_path+'//src//'+entity_file + '.txt', dtype=np.int32)
        np.save(file_path+'//'+entity_file + '.npy', entity_np)

    # ####这里对要推荐的患者的疾病和症状进行读取，因为报错，先手工处理一下
    # if os.path.exists(predict_file + '.npy'):
    #     predict_np = np.load(predict_file + '.npy')
    # else:
    #     predict_np = np.loadtxt(predict_file + '.txt', dtype=np.int32)
    #     np.save(predict_file + '.npy', predict_np)

    # combine_rating_np=np.append(train_rating_np,predict_rating_np)   ###这个是“患者_食品_是否”文件的合并
    # combine_entity_predict_np=np.append(entity_np,predict_np)      ###这个是“患者_疾病症状_是否”文件的合并
    #logging.info("contructing users' initial entity set ...")

    user_history_item_dict = dict()
    item_history_user_dict = dict()
    item_neighbor_item_dict = dict()
    for i in range(entity_np.shape[0]):
        user = entity_np[i][0]
        item = entity_np[i][1]
        rating = entity_np[i][2]
        if rating == 1:
            if user not in user_history_item_dict:
                user_history_item_dict[user] = []
            user_history_item_dict[user].append(item)
    # print(train_rating_np[:, 1])
    # print(type(train_rating_np[:, 1]))
    # item_list_1 = train_rating_np[:, 1]
    ##这里先不合并预测患者的疾病症状，因为后期患者疾病症状如果没在知识图谱中，也不会放进训练模型
    item_list_2 = predict_rating_np[:, 1]
    item_list=set(item_list_2)
    # print(np.append(item_list_1,item_list_2))
    # print(type(np.append(item_list_1,item_list_2)))
    for item in item_list:
        if item not in item_neighbor_item_dict:
            item_neighbor_item_dict[item] = [item]
    return user_history_item_dict, item_neighbor_item_dict


def load_kg(args):
    kg_file = 'kg_dataset_20220626'
    logging.info("loading kg file: %s.npy", kg_file)
    file_path = os.path.dirname(__file__)
    if os.path.exists(file_path +'\\src\\'+ kg_file + '.npy'):
        kg_np = np.load(file_path +'\\src\\'+ kg_file + '.npy')
    else:
        kg_np = np.loadtxt(file_path +'\\src\\'+ kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    n_entity = 7572    ###三元组有多少行就设置成多少
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg, kg_np


def construct_kg(kg_np):
    #logging.info("constructing knowledge graph ...")
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def kg_propagation(args, kg, init_entity_set, set_size, is_user):
    triple_sets = collections.defaultdict(list)
    for obj in init_entity_set.keys():
        if is_user and args.n_layer == 0:
            n_layer = 1
        else:
            n_layer = args.n_layer
        for l in range(n_layer):
            h,r,t = [],[],[]
            if l == 0:
                entities = init_entity_set[obj]
            else:
                entities = triple_sets[obj][-1][2]
            for entity in entities:
                for tail_and_relation in kg[entity]:
                    h.append(entity)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])
            # if len(h) == 0:
            #     triple_sets[obj].append(triple_sets[obj][-1])
            # else:
            if len(h)!=0:
                indices = np.random.choice(len(h), size=set_size, replace= (len(h) < set_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
                triple_sets[obj].append((h, r, t))
    return triple_sets
