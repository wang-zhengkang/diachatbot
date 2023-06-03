import json
import os
import random
import sys
import zipfile
import numpy
import torch
from pprint import pprint
from tqdm import tqdm

def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        predicts = [[x[0], x[1], x[2], x[3].lower()] for x in predicts]
        labels = item['golden']
        labels = [[x[0], x[1], x[2], x[3].lower()] for x in labels]
        for ele in predicts:
            if ele in labels:
                TP += 1
            else:
                FP += 1
        for ele in labels:
            if ele not in predicts:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, F1

def load_data(key,mode):
    assert mode == 'All' or mode == 'User' or mode == 'Doctor'
    assert key == 'train' or key == 'test' or key == 'val'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(cur_dir, 'data/{}_data'.format(mode))
    with open(os.path.join(processed_data_dir, '{}_data.json'.format(key)),encoding='utf-8') as f:
        data = json.load(f)
    get_data = {}
    get_data['utterance'] = []
    get_data['dialog_act'] = []
    for utterances in data:
        sentance = ""
        for word in utterances[0]:
            sentance += word
        get_data['utterance'].append(sentance)
        get_data['dialog_act'].append(utterances[3])
    return get_data

# if __name__ == '__main__':
#     data = load_data(key="test", mode="All")
#     print(data)


if __name__ == '__main__':
    seed = 2020
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    dataset_name = 'data'
    model_name = 'BERTNLU'

    if dataset_name == 'data':
        if model_name == 'BERTNLU':
            from convlab2.nlu.jointBERT.diachat.nlu import BERTNLU
            model = BERTNLU()
        else:
            raise Exception("Available models: BERTNLU")

        data = load_data(key = "test",mode = "User")
        predict_golden = []

        # tqdm 进度条
        for i in tqdm(range(len(data['utterance']))):
            predict = model.predict(utterance=data['utterance'][i])
            label = data['dialog_act'][i]
            predict_golden.append({
                'predict': predict,
                'golden': label
            })

        precision, recall, F1 = calculateF1(predict_golden)
        print('Model {} on {} {} sentences:'.format(model_name, dataset_name, len(predict_golden)))
        print('\t Precision: %.2f' % (100 * precision))
        print('\t Recall: %.2f' % (100 * recall))
        print('\t F1: %.2f' % (100 * F1))

    else:
        raise Exception("currently supported dataset: data")
