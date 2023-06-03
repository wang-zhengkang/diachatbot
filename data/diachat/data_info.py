'''
获取diachat数据集信息(待完善)
使用方法:在同一个文件夹下修改data_source_name即可
'''
import json
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
data_source_name = "annotations_20220627_2.json"
data_source = os.path.join(current_dir, data_source_name)

with open(data_source, encoding="utf-8") as data_source:
    data_source = json.load(data_source)
    data = []
    utterance_num = 0
    for i, conversation in enumerate(data_source):
        utterance_num += len(conversation['utterances'])
    print(utterance_num)