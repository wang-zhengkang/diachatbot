import numpy as np
import torch
import torch.nn as nn 
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from diachatbot.ai.ckan.CKAN_model import CKAN
import logging
from sklearn.metrics import classification_report
import random
import joblib

#logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)
device = torch.device("cpu")

def sample_pos_triples_for_h(kg_dict, head, n_sample_pos_triples):
    pos_triples = kg_dict[head]
    n_pos_triples = len(pos_triples)

    sample_relations, sample_pos_tails = [], []
    while True:
        if len(sample_relations) == n_sample_pos_triples:
            break
        pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
        tail = pos_triples[pos_triple_idx][0]
        relation = pos_triples[pos_triple_idx][1]
        if relation not in sample_relations and tail not in sample_pos_tails:
            sample_relations.append(relation)
            sample_pos_tails.append(tail)
    return sample_relations, sample_pos_tails


def sample_neg_triples_for_h(kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
    pos_triples = kg_dict[head]
    sample_neg_tails = []
    while True:
        if len(sample_neg_tails) == n_sample_neg_triples:
            break
        tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
        if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
            sample_neg_tails.append(tail)
    return sample_neg_tails

def generate_kg_batch(kg_dict, batch_size, highest_neg_idx):
    exist_heads = list(kg_dict.keys())

    if batch_size <= len(exist_heads):
        batch_head = random.sample(exist_heads, batch_size)
    else:
        batch_head = [random.choice(exist_heads) for _ in range(batch_size)]
    batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
    for h in batch_head:
        relation, pos_tail = sample_pos_triples_for_h(kg_dict, h, 1)
        batch_relation += relation
        batch_pos_tail += pos_tail
        neg_tail = sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
        batch_neg_tail += neg_tail
    batch_head = torch.LongTensor(batch_head)
    batch_relation = torch.LongTensor(batch_relation)
    batch_pos_tail = torch.LongTensor(batch_pos_tail)
    batch_neg_tail = torch.LongTensor(batch_neg_tail)
    return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


def train(args, data_info):
    #logging.info("================== training CKAN ====================")
    test_data = data_info[0]
    user_triple_set = data_info[5]
    item_triple_set = data_info[6]
    n_entity = data_info[3]
    n_relation = data_info[4]
    kg = data_info[7]
    kg_np = data_info[8]
    model = CKAN(args, n_entity, n_relation)
    if args.use_cuda:
        model.to(device)
    cf_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
    )
    kg_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
    )
    for step in range(args.n_epoch):
        np.random.shuffle(train_data)
        start_cf = 0
        start_kg = 0
        
        model.train()
        while start_kg < len(kg_np):
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = generate_kg_batch(kg,args.kg_batch_size,n_entity)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)
            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')
            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            start_kg += args.kg_batch_size

        while start_cf < train_data.shape[0]:
            labels = _get_feed_label(args, train_data[start_cf:start_cf + args.batch_size, 2])
            cf_batch_loss = model(labels,*_get_feed_data(args, train_data, user_triple_set, item_triple_set, start_cf, start_cf + args.batch_size), mode='train_cf')
            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            start_cf += args.batch_size

        model_name='CKAN_epoch'+str(step+1)

        eval_auc, eval_f1, eval_pre, eval_recall = ctr_eval(args, model, eval_data, user_triple_set, item_triple_set)
        test_auc, test_f1, test_pre, test_recall = ctr_eval(args, model, test_data, user_triple_set, item_triple_set)
        ctr_info = 'epoch %.2d    eval auc: %.4f f1: %.4f  pre: %.4f recall: %.4f  test auc: %.4f f1: %.4f pre: %.4f recall: %.4f'
        #logging.info(ctr_info, step, eval_auc, eval_f1, eval_pre, eval_recall, test_auc, test_f1, test_pre, test_recall)
        # joblib.dump(model,'../saved_model/'+model_name+'.pkl')
        # print('Model Saved')
        if args.show_topk:
            topk_eval(args, model, train_data, test_data, user_triple_set, item_triple_set)



def ctr_eval(args, model, data, user_triple_set, item_triple_set):
    model.eval()
    start = 0
    y_patient_id = []
    y_food_entity_id = []
    y_true_list = []
    y_score_list = []
    y_pred_list = []
    # print(data[start:start + args.batch_size,0])
    while start < data.shape[0]:
        patient_id = data[start:start + args.batch_size, 0]
        food_entity_id = data[start:start + args.batch_size, 1]
        labels = data[start:start + args.batch_size, 2]
        scores = model(*_get_feed_data(args, data, user_triple_set, item_triple_set, start, start + args.batch_size), mode='predict')
        scores = scores.detach().cpu().numpy()
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        y_patient_id += patient_id.tolist()
        y_food_entity_id += food_entity_id.tolist()
        y_true_list += labels.tolist()
        y_score_list += scores.tolist()
        y_pred_list += predictions
        start += args.batch_size
    model.train()  
    # auc = roc_auc_score(y_true_list,y_score_list)
    # f1 = f1_score(y_true_list,y_pred_list)
    # pre = precision_score(y_true_list,y_pred_list)
    # recall = recall_score(y_true_list,y_pred_list)
    # print(y_patient_id,y_food_entity_id,y_true_list,y_score_list,y_pred_list)
    return y_patient_id,y_food_entity_id,y_true_list,y_score_list,y_pred_list

def DCG(A, test_set):
    dcg = 0
    r_i = 0
    if A[0] in test_set:
        r_i = 1
    dcg = r_i
    for i in range(1,len(A)):
        r_i = 0
        if A[i] in test_set:
            r_i = 1
        dcg += r_i / np.log2(i+1)
    return dcg


def topk_eval(args, model, train_data, test_data, user_triple_set, item_triple_set):
    # logging.info('calculating recall ...')
    k_list = [5, 10, 20]
    recall_list = {k: [] for k in k_list}
    dcg_list = {k: [] for k in k_list}
    item_set = set(train_data[:,1].tolist() + test_data[:,1].tolist())
    train_record = _get_user_record(args, train_data, True)
    test_record = _get_user_record(args, test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if user_list == []:
        user_list = list(set(test_record.keys()))
    user_num = 500
    if len(user_list) > user_num:
        np.random.seed()    
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    test_item_user = dict()
    for i in range(len(test_data)):
        if test_data[i,0] not in test_item_user:
            test_item_user[test_data[i,0]] = []
        test_item_user[test_data[i,0]].append(test_data[i,1])
    model.eval()
    for user in user_list:
        if user not in train_record:
            test_item_list = test_item_user[user]
        else:
            test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size] 
            input_data = _get_topk_feed_data(user, items)
            scores = model(*_get_feed_data(args, input_data, user_triple_set, item_triple_set, 0, args.batch_size), mode='predict')
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            scores = model(*_get_feed_data(args, input_data, user_triple_set, item_triple_set, 0, args.batch_size), mode='predict')
            res_items = test_item_list[start:]
            scores = scores[start:start+len(test_item_list)]
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(item_sorted[:k])))
            dcg_list[k].append(DCG(item_sorted[:k],test_record[user]))
    model.train()  
    recall = [np.mean(recall_list[k]) for k in k_list]
    dcg = [np.mean(dcg_list[k]) for k in k_list]
    _show_recall_info(zip(k_list, recall,dcg))

    
    
def _get_feed_data(args, data, user_triple_set, item_triple_set, start, end):
    items = torch.LongTensor(data[start:end, 1])
    if args.use_cuda:
        items = items.to(device)
    users_triple = _get_triple_tensor(args, data[start:end,0], user_triple_set)
    items_triple = _get_triple_tensor(args, data[start:end,1], item_triple_set)
    return items, users_triple, items_triple


def _get_feed_label(args,labels):
    labels = torch.FloatTensor(labels)
    if args.use_cuda:
        labels = labels.to(device)
    return labels


def _get_triple_tensor(args, objs, triple_set):
    h,r,t = [], [], []
    for i in range(args.n_layer):
        h.append(torch.LongTensor([triple_set[obj][i][0] for obj in objs]))
        r.append(torch.LongTensor([triple_set[obj][i][1] for obj in objs]))
        t.append(torch.LongTensor([triple_set[obj][i][2] for obj in objs]))
        if args.use_cuda:
            h = list(map(lambda x: x.to(device), h))
            r = list(map(lambda x: x.to(device), r))
            t = list(map(lambda x: x.to(device), t))
    return [h,r,t]


def _get_user_record(args, data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user,item])
    return np.array(res)


def _show_recall_info(recall_zip):
    res = ""
    for i,j,k in recall_zip:
        res += "K@%d:%.4f  DCG@%d:%.4f"%(i,j,i,k)
    #logging.info(res)



