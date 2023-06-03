#!/usr/bin/env python
# coding: utf-8

# # Generate training data

import os
import json
import jieba
import re
from pprint import pprint
from collections import defaultdict
from copy import copy
import random
import zipfile
import functools

from convlab2.util.diachat.domain_act_slot import *

# act_labels={
#     "noargs_act":['Accept','AskWhy','Assure','Deny','GeneralAdvice','Chitchat','GeneralExplanation'],
#     "doctor_act":['Advice','AdviceNot','Explanation','AskForSure','AskFor','AskHow','Assure','Deny','GeneralAdvice','Chitchat','GeneralExplanation'],
#     "user_act":['Inform','Accept','AskForSure','AskWhy','AskFor','AskHow','Assure','Deny','Uncertain','Chitchat'],
#     "all_act":['Inform','Advice','AdviceNot','Accept','Explanation','AskForSure','AskWhy','AskFor','AskHow','Assure','Deny','Uncertain','GeneralAdvice','Chitchat','GeneralExplanation']
#     }
intent_order = {
    'User': (
        'Accept++',
        'AskFor+基本信息+既往史',
        'AskFor+治疗+持续时长',
        'AskFor+治疗+效果',
        'AskFor+治疗+时间',
        'AskFor+治疗+检查值',
        'AskFor+治疗+检查项',
        'AskFor+治疗+治疗名',
        'AskFor+治疗+用药量',
        'AskFor+治疗+用药（治疗）频率',
        'AskFor+治疗+药品',
        'AskFor+治疗+药品类型',
        'AskFor+治疗+适应症',
        'AskFor+治疗+部位',
        'AskFor+行为+持续时长',
        'AskFor+行为+效果',
        'AskFor+行为+行为名',
        'AskFor+行为+频率',
        'AskFor+运动+持续时长',
        'AskFor+运动+时间',
        'AskFor+运动+运动名',
        'AskFor+问题+持续时长',
        'AskFor+问题+时间',
        'AskFor+问题+状态',
        'AskFor+问题+疾病',
        'AskFor+问题+症状',
        'AskFor+问题+症状部位',
        'AskFor+问题+血糖值',
        'AskFor+饮食+成份量',
        'AskFor+饮食+成分',
        'AskFor+饮食+效果',
        'AskFor+饮食+时间',
        'AskFor+饮食+饮食名',
        'AskFor+饮食+饮食量',
        'AskForSure+基本信息+体重',
        'AskForSure+基本信息+年龄',
        'AskForSure+基本信息+性别',
        'AskForSure+基本信息+既往史',
        'AskForSure+基本信息+身高',
        'AskForSure+治疗+持续时长',
        'AskForSure+治疗+效果',
        'AskForSure+治疗+时间',
        'AskForSure+治疗+检查值',
        'AskForSure+治疗+检查项',
        'AskForSure+治疗+治疗名',
        'AskForSure+治疗+用药量',
        'AskForSure+治疗+用药（治疗）频率',
        'AskForSure+治疗+药品',
        'AskForSure+治疗+药品类型',
        'AskForSure+治疗+适应症',
        'AskForSure+治疗+部位',
        'AskForSure+行为+效果',
        'AskForSure+行为+时间',
        'AskForSure+行为+行为名',
        'AskForSure+行为+频率',
        'AskForSure+运动+强度',
        'AskForSure+运动+持续时长',
        'AskForSure+运动+效果',
        'AskForSure+运动+时间',
        'AskForSure+运动+运动名',
        'AskForSure+运动+频率',
        'AskForSure+问题+持续时长',
        'AskForSure+问题+时间',
        'AskForSure+问题+状态',
        'AskForSure+问题+疾病',
        'AskForSure+问题+症状',
        'AskForSure+问题+症状部位',
        'AskForSure+问题+血糖值',
        'AskForSure+饮食+成份量',
        'AskForSure+饮食+成分',
        'AskForSure+饮食+效果',
        'AskForSure+饮食+时间',
        'AskForSure+饮食+饮食名',
        'AskForSure+饮食+饮食量',
        'AskHow++',
        'AskHow+基本信息+体重',
        'AskHow+基本信息+性别',
        'AskHow+治疗+时间',
        'AskHow+治疗+检查值',
        'AskHow+治疗+检查项',
        'AskHow+治疗+治疗名',
        'AskHow+治疗+用药量',
        'AskHow+治疗+药品',
        'AskHow+治疗+部位',
        'AskHow+行为+效果',
        'AskHow+行为+时间',
        'AskHow+行为+行为名',
        'AskHow+运动+持续时长',
        'AskHow+运动+运动名',
        'AskHow+问题+持续时长',
        'AskHow+问题+时间',
        'AskHow+问题+状态',
        'AskHow+问题+疾病',
        'AskHow+问题+症状',
        'AskHow+问题+症状部位',
        'AskHow+问题+血糖值',
        'AskHow+饮食+效果',
        'AskHow+饮食+饮食名',
        'AskHow+饮食+饮食量',
        'AskWhy++',
        'Assure++',
        'Chitchat++',
        'Deny++',
        'Inform+基本信息+体重',
        'Inform+基本信息+年龄',
        'Inform+基本信息+性别',
        'Inform+基本信息+既往史',
        'Inform+基本信息+身高',
        'Inform+治疗+持续时长',
        'Inform+治疗+效果',
        'Inform+治疗+时间',
        'Inform+治疗+检查值',
        'Inform+治疗+检查项',
        'Inform+治疗+治疗名',
        'Inform+治疗+用药量',
        'Inform+治疗+用药（治疗）频率',
        'Inform+治疗+药品',
        'Inform+治疗+药品类型',
        'Inform+治疗+适应症',
        'Inform+治疗+部位',
        'Inform+行为+持续时长',
        'Inform+行为+效果',
        'Inform+行为+时间',
        'Inform+行为+行为名',
        'Inform+行为+频率',
        'Inform+运动+强度',
        'Inform+运动+持续时长',
        'Inform+运动+效果',
        'Inform+运动+时间',
        'Inform+运动+运动名',
        'Inform+运动+频率',
        'Inform+问题+持续时长',
        'Inform+问题+时间',
        'Inform+问题+状态',
        'Inform+问题+疾病',
        'Inform+问题+症状',
        'Inform+问题+症状部位',
        'Inform+问题+血糖值',
        'Inform+饮食+成份量',
        'Inform+饮食+成分',
        'Inform+饮食+时间',
        'Inform+饮食+饮食名',
        'Inform+饮食+饮食量',
        'Uncertain++',
        'Uncertain+行为+行为名',
        'Uncertain+问题+症状'
    ),
    'Doctor': (
        'Advice+基本信息+体重',
        'Advice+治疗+持续时长',
        'Advice+治疗+效果',
        'Advice+治疗+时间',
        'Advice+治疗+检查值',
        'Advice+治疗+检查项',
        'Advice+治疗+治疗名',
        'Advice+治疗+用药量',
        'Advice+治疗+用药（治疗）频率',
        'Advice+治疗+药品',
        'Advice+治疗+药品类型',
        'Advice+治疗+适应症',
        'Advice+治疗+部位',
        'Advice+行为+持续时长',
        'Advice+行为+效果',
        'Advice+行为+时间',
        'Advice+行为+行为名',
        'Advice+行为+频率',
        'Advice+运动+强度',
        'Advice+运动+持续时长',
        'Advice+运动+效果',
        'Advice+运动+时间',
        'Advice+运动+运动名',
        'Advice+运动+频率',
        'Advice+问题+时间',
        'Advice+问题+状态',
        'Advice+问题+疾病',
        'Advice+问题+症状',
        'Advice+问题+症状部位',
        'Advice+问题+血糖值',
        'Advice+饮食+成份量',
        'Advice+饮食+成分',
        'Advice+饮食+效果',
        'Advice+饮食+时间',
        'Advice+饮食+饮食名',
        'Advice+饮食+饮食量',
        'AdviceNot+治疗+检查值',
        'AdviceNot+治疗+检查项',
        'AdviceNot+治疗+治疗名',
        'AdviceNot+治疗+用药量',
        'AdviceNot+治疗+药品',
        'AdviceNot+治疗+药品类型',
        'AdviceNot+治疗+部位',
        'AdviceNot+行为+持续时长',
        'AdviceNot+行为+时间',
        'AdviceNot+行为+行为名',
        'AdviceNot+行为+频率',
        'AdviceNot+运动+强度',
        'AdviceNot+运动+时间',
        'AdviceNot+运动+运动名',
        'AdviceNot+问题+时间',
        'AdviceNot+问题+疾病',
        'AdviceNot+问题+症状',
        'AdviceNot+问题+血糖值',
        'AdviceNot+饮食+成份量',
        'AdviceNot+饮食+成分',
        'AdviceNot+饮食+时间',
        'AdviceNot+饮食+饮食名',
        'AdviceNot+饮食+饮食量',
        'AskFor+基本信息+体重',
        'AskFor+基本信息+年龄',
        'AskFor+基本信息+性别',
        'AskFor+基本信息+既往史',
        'AskFor+基本信息+身高',
        'AskFor+治疗+持续时长',
        'AskFor+治疗+时间',
        'AskFor+治疗+检查值',
        'AskFor+治疗+检查项',
        'AskFor+治疗+用药量',
        'AskFor+治疗+用药（治疗）频率',
        'AskFor+治疗+药品',
        'AskFor+治疗+药品类型',
        'AskFor+治疗+部位',
        'AskFor+行为+持续时长',
        'AskFor+行为+效果',
        'AskFor+行为+时间',
        'AskFor+行为+行为名',
        'AskFor+行为+频率',
        'AskFor+运动+时间',
        'AskFor+运动+运动名',
        'AskFor+问题+持续时长',
        'AskFor+问题+时间',
        'AskFor+问题+疾病',
        'AskFor+问题+症状',
        'AskFor+问题+症状部位',
        'AskFor+问题+血糖值',
        'AskFor+饮食+时间',
        'AskFor+饮食+饮食名',
        'AskFor+饮食+饮食量',
        'AskForSure+基本信息+体重',
        'AskForSure+基本信息+性别',
        'AskForSure+基本信息+既往史',
        'AskForSure+治疗+效果',
        'AskForSure+治疗+时间',
        'AskForSure+治疗+检查值',
        'AskForSure+治疗+检查项',
        'AskForSure+治疗+治疗名',
        'AskForSure+治疗+药品',
        'AskForSure+治疗+药品类型',
        'AskForSure+治疗+部位',
        'AskForSure+行为+效果',
        'AskForSure+行为+时间',
        'AskForSure+行为+行为名',
        'AskForSure+行为+频率',
        'AskForSure+运动+强度',
        'AskForSure+运动+时间',
        'AskForSure+运动+运动名',
        'AskForSure+运动+频率',
        'AskForSure+问题+时间',
        'AskForSure+问题+疾病',
        'AskForSure+问题+症状',
        'AskForSure+问题+症状部位',
        'AskForSure+问题+血糖值',
        'AskForSure+饮食+时间',
        'AskForSure+饮食+饮食名',
        'AskForSure+饮食+饮食量',
        'AskHow++',
        'AskHow+治疗+时间',
        'AskHow+治疗+检查值',
        'AskHow+治疗+检查项',
        'AskHow+治疗+药品',
        'AskHow+行为+时间',
        'AskHow+行为+行为名',
        'AskHow+问题+状态',
        'AskHow+问题+症状',
        'AskHow+问题+症状部位',
        'Assure++',
        'Chitchat++',
        'Deny++',
        'Explanation+基本信息+体重',
        'Explanation+基本信息+年龄',
        'Explanation+基本信息+性别',
        'Explanation+基本信息+既往史',
        'Explanation+治疗+持续时长',
        'Explanation+治疗+效果',
        'Explanation+治疗+时间',
        'Explanation+治疗+检查值',
        'Explanation+治疗+检查项',
        'Explanation+治疗+治疗名',
        'Explanation+治疗+用药量',
        'Explanation+治疗+用药（治疗）频率',
        'Explanation+治疗+药品',
        'Explanation+治疗+药品类型',
        'Explanation+治疗+适应症',
        'Explanation+治疗+部位',
        'Explanation+行为+持续时长',
        'Explanation+行为+效果',
        'Explanation+行为+时间',
        'Explanation+行为+行为名',
        'Explanation+行为+频率',
        'Explanation+运动+强度',
        'Explanation+运动+持续时长',
        'Explanation+运动+运动名',
        'Explanation+运动+频率',
        'Explanation+问题+持续时长',
        'Explanation+问题+时间',
        'Explanation+问题+状态',
        'Explanation+问题+疾病',
        'Explanation+问题+症状',
        'Explanation+问题+症状部位',
        'Explanation+问题+血糖值',
        'Explanation+饮食+成份量',
        'Explanation+饮食+成分',
        'Explanation+饮食+效果',
        'Explanation+饮食+时间',
        'Explanation+饮食+饮食名',
        'Explanation+饮食+饮食量',
        'GeneralAdvice++',
        'GeneralAdvice+治疗+',
        'GeneralAdvice+行为+',
        'GeneralAdvice+运动+',
        'GeneralAdvice+问题+',
        'GeneralAdvice+饮食+',
        'GeneralExplanation++',
        'GeneralExplanation+基本信息+',
        'GeneralExplanation+治疗+',
        'GeneralExplanation+行为+',
        'GeneralExplanation+运动+',
        'GeneralExplanation+问题+',
        'GeneralExplanation+饮食+'
    )
}
intents = {
    "User":[],
    "Doctor":[]
    }

def meger_intent(role,intent):
    if intent not in intents[role]:
        intents[role].append(intent)

def meger_intent2(role):
    for intnt in intent_order[role]:
        meger_intent(role,intnt)

def act_no_args(act):
    return act in act_labels['noargs_act']

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def main():
    def cmp_intent(intent1: str, intent2: str):
        assert role in ['Doctor', 'User']

        intent1 = intent1.split('1')[0]
        intent2 = intent2.split('1')[0]
#         if 'Inform' in intent1 and '无' in intent1:
#             intent1 = 'Inform+主体+属性+无'
#         if 'Inform' in intent2 and '无' in intent2:
#             intent2 = 'Inform+主体+属性+无'
        try:
            assert intent1 in intent_order[role] and intent2 in intent_order[role]
        except AssertionError:
            print(role,intent1, intent2)
            #add by yjf
            meger_intent(role,intent1)
            meger_intent(role,intent2)
        return intent_order[role].index(intent1) - intent_order[role].index(intent2)

    data_dir = '../../../../../data/diachat'
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), data_dir))
    train_archive = zipfile.ZipFile(os.path.join(data_dir, 'train.json.zip'), 'r')
    train_data = json.load(train_archive.open('train.json'))
    valid_archive = zipfile.ZipFile(os.path.join(data_dir, 'val.json.zip'), 'r')
    valid_data = json.load(valid_archive.open('val.json'))
    test_archive = zipfile.ZipFile(os.path.join(data_dir, 'test.json.zip'), 'r')
    test_data = json.load(test_archive.open('test.json'))

    data = {'train': train_data, 'valid': valid_data, 'test': test_data}

    print("Length of train_data:", len(train_data))

    # ## For system/user

    user_multi_intent_dict = defaultdict(list)
    sys_multi_intent_dict = defaultdict(list)

    role = None
    dialogue_id = 1
    for dialogue in train_data:# dialogue是字典  只处理了training data
        # print('Processing the %dth dialogue' % dialogue_id)
        dialogue_id += 1
        for round in dialogue['utterances']:
            # original content
            content = round['utterance']
            intent_list = []
            intent_frequency = defaultdict(int)
            role = round['agentRole']
            usable = True
            for act in round['annotation']:
                cur_act = copy(act)

                for ele in act['slot_values']:#slot_values对应的列表中可能有多个字典
                    intent = '+'.join([act['act_label'],ele['domain'],ele['slot']])
                    intent_list.append(intent)
                    intent_frequency[intent] += 1
                    
                    value_pos = ele['pos']
                    if len(value_pos) > 0:
                        # value to be replaced
                        value = ele['value']

                        # placeholder
                        placeholder = '[' + intent + ']'
                        placeholder_one = '[' + intent + '1]'
                        placeholder_with_number = '[' + intent + str(intent_frequency[intent]) + ']'

                        if intent_frequency[intent] > 1:
                            content = content.replace(placeholder, placeholder_one)
                            content = content.replace(value, placeholder_with_number)
                        else:
                            content = content.replace(value, placeholder)
#                         else:
#                             usable = False

                # multi-intent name
            #modified by houxintong, shift a tab left    
            try:
                intent_list = sorted(intent_list, key=functools.cmp_to_key(cmp_intent))
            except:
#                     print(round['utterance'])
                pass
            multi_intent = '*'.join(intent_list)
            if usable:
                if round['agentRole'] == 'User':
                    user_multi_intent_dict[multi_intent].append(content)
                else:
                    sys_multi_intent_dict[multi_intent].append(content)
    print('运行成功1')
    #输出排序表
    if len(intents['User']) > 0:
        print("User intents:")
        meger_intent2('User')
        intents['User'].sort()
        for intnt in intents['User']:
            print("'{}',".format(intnt))
    if len(intents['Doctor']) > 0:        
        print("Doctor intents:")
        meger_intent2('Doctor')
        intents['Doctor'].sort()
        for intnt in intents['Doctor']:
            print("'{}',".format(intnt))



    output_data_dir = 'resource'
    for require_role in ['Doctor', 'User']:
        print('\nProcessing %s data...' % require_role)
        if require_role == 'User':
            output_data_dir += '_usr'
        if not os.path.exists(output_data_dir):
            os.mkdir(output_data_dir)
        if require_role == 'User':
            template_data = user_multi_intent_dict.copy()
        else:
            template_data = sys_multi_intent_dict.copy()
        print('Number of intents in templates:', len(template_data))

        sens = []
        for key, ls in template_data.items():
            if key:
                sens += ls
        print('Number of sentences in templates:', len(sens))

        # ### vocab.txt

        vocab_dict = defaultdict(int)
        pattern = re.compile(r'(\[[^\[^\]]+\])')
        for ls in template_data.values():
            for sen in ls:
                slots = pattern.findall(sen)
                for slot in slots:
                    vocab_dict[slot] += 1
                    sen = sen.replace(slot, '')
                for word in jieba.lcut(sen):
                    vocab_dict[word] += 1
        len(vocab_dict)

        vocab_dict = {word: frequency for word, frequency in vocab_dict.items() if vocab_dict[word] > 0} # 原来是>3
        len(vocab_dict)

        with open(os.path.join(output_data_dir, 'vocab.txt'), 'w', encoding='utf-8') as fvocab:
            fvocab.write('PAD_token\nSOS_token\nEOS_token\nUNK_token\n')
            for key, value in sorted(vocab_dict.items(), key=lambda x: int(x[1])):
                if key.strip():
                    fvocab.write(key + '\t' + str(value) + '\n')

        # ### text.json, feat.json
        print('运行成功2')

        def split_delex_sentence(sen):
            ori_sen = copy(sen)
            res_sen = ''
            pattern = re.compile(r'(\[[^\[^\]]+\])')
            slots = pattern.findall(sen)
            for slot in slots:
                sen = sen.replace(slot, '[slot]')
            sen = sen.split('[slot]')
            for part in sen:
                part = ' '.join(jieba.lcut(part))
                res_sen += part
                if slots:
                    res_sen += ' ' + slots.pop(0) + ' '
            return res_sen

        dialogue_id = 0
        text_dict = {'train': defaultdict(dict), 'valid': defaultdict(dict), 'test': defaultdict(dict)}
        all_text_dict = defaultdict(dict)
        feat_dict = defaultdict(dict)
        template_list = []
        unk_sen_num = []
        print('运行成功3')

        for split in ['train', 'valid', 'test']:
            for dialogue in data[split]:
                dialogue_id += 1
                if (dialogue_id % 100 == 0):
                    print('Processing the %dth dialogue' % dialogue_id)
                round_id = 0
                for round in dialogue['utterances']:
                    # original content
                    content = round['utterance']
                    ori_content = content
                    intent_list = []
                    intent_frequency = defaultdict(int)
                    role = round['agentRole']

                    # now we consider the system/user:
                    if role != require_role:
                        continue
                    round_id += 1


                    # usable = True
                    for act in round['annotation']:
                        cur_act = copy(act)

                        for ele in cur_act['slot_values']:
                            # slot_values对应的列表中可能有多个字典
                            # content replacement
                            value = 'none'
                            freq = 'none'
                            value = 'none'
                            intent = '+'.join([cur_act['act_label'], ele['domain'], ele['slot']])
                            intent_list.append(intent)
                            intent_frequency[intent] += 1
                            
                            value_pos = ele['pos']
                            if len(value_pos) > 0:
                                
                                value = ele['value']
                                # placeholder
                                placeholder = '[' + intent + ']'
                                placeholder_one = '[' + intent + '1]'
                                placeholder_with_number = '[' + intent + str(intent_frequency[intent]) + ']'
    
                                if intent_frequency[intent] > 1:
                                    content = content.replace(placeholder, placeholder_one)
                                    content = content.replace(value, placeholder_with_number)
                                else:
                                    content = content.replace(value, placeholder)
    
                                freq = str(intent_frequency[intent])

                            if cur_act['act_label'] == 'AskFor' and len(value_pos) == 0:
                                value='?'
                                freq = '?'
                                
                            new_act = intent.split('+')
#                             feat_value = [fslot, freq, value]
                            if act_no_args(new_act[0]):#如果act 没有arguments
                                if new_act[0] in ['GeneralAdvice','GeneralExplanation']:
                                    feat_key = new_act[1] + '-' + new_act[0]
                                    feat_value = ['none','none','none']
                                else:                                
                                    feat_key = 'General-'+new_act[0]
                                    feat_value = ['none','none','none']
                            else:
                                feat_key = new_act[1] + '-' + new_act[0]
                                feat_value = [new_act[2], freq, value]
                            
#                             fslot = 'none' if new_act[2] == '' else new_act[2]
#                             feat_value = [fslot, freq, value]

                            feat_dict[dialogue['conversationId']][round_id] = feat_dict[dialogue['conversationId']].get(round_id, dict())
                            feat_dict[dialogue['conversationId']][round_id][feat_key] = feat_dict[dialogue['conversationId']][round_id].get(feat_key, [])
                            feat_dict[dialogue['conversationId']][round_id][feat_key].append(feat_value)

                        # save to text.json
                        split_delex = split_delex_sentence(content)
                        unk_sen = [word if word in vocab_dict else 'UNK_token' for word in
                               re.split(r'\s+', split_delex) if word]
                        unk_sen_num.append('UNK_token' in unk_sen)
                        text_dict[split][dialogue['conversationId']][round_id] = {
                            "delex": split_delex,
                            "ori": ori_content
                        }
                        all_text_dict[dialogue['conversationId']][round_id] = {
                            "delex": split_delex,
                            "ori": ori_content
                        }
        print('unk sen ratio', sum(unk_sen_num)*1.0/(len(unk_sen_num)))

        with open(os.path.join(output_data_dir, 'text.json'), 'w', encoding='utf-8') as f:
            json.dump(all_text_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

        with open(os.path.join(output_data_dir, 'feat.json'), 'w', encoding='utf-8') as f:
            json.dump(feat_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

        # ### template.txt

        template_set = set()
        for dialogue in feat_dict.values():
            for r in dialogue.values():
                for k, ls in r.items():
                    template_set.add('d:' + k.split('-')[0])
                    template_set.add('d-a:' + k)
                    for v in ls:
                        template_set.add('d-a-s-v:' + k + '-' + v[0] + '-' + str(v[1]))
        with open(os.path.join(output_data_dir, 'template.txt'), 'w', encoding='utf-8') as ftem:
            ftem.write('\n'.join(sorted(list(template_set), reverse=True)))

        # ### split data

        split_dict = {'valid': [], 'test': [], 'train': []}
        for split in split_dict.keys():
            all_candidates = []
            for d_id, sens in text_dict[split].items():
                for i in range(len(sens)):
                    all_candidates.append([d_id, str(i + 1), "-"])
            split_dict[split] = copy(all_candidates)

        with open(os.path.join(output_data_dir, 'split.json'), 'w', encoding='utf-8') as f:
            json.dump(split_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == '__main__':
    main()
