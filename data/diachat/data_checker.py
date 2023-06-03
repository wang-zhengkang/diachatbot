'''
yangjinfeng
'''
import json
from convlab2.util.diachat.domain_act_slot import *

tag_replace_dict={
    '"ask"':'"AskFor"',
    '"Ask-for"':'"AskFor"',
    '"Ask-for-sure"':'"AskForSure"',
    '"Ask-why"':'"AskWhy"',
    '"Ask-how"':'"AskHow"',
    '"inform"':'"Inform"',
    '"Advice-not"':'"AdviceNot"',
    '"chichat"':'"Chitchat"',
    '"General-explanation"':'"GeneralExplanation"',
    '"General-advice"':'"GeneralAdvice"',
    '•':'.', #特殊符号替换
    '·':'.'
    }

BubianQuanjiao = '，？；：！（）'



"""全角转半角"""
def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        if uchar in BubianQuanjiao:
            rstring += uchar
            continue
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


'''
    先做全角转半角，然后做act label替换
'''
def replace_line(line):
    result = strQ2B(line)
    for tag in tag_replace_dict:
        result = result.replace(tag,tag_replace_dict[tag])
    return result

def replaceFile(input,output):
    replaced = []
    with open(input, 'r',encoding='utf-8') as f:
        for line in f:
            replaced += replace_line(line)
    
    with open(output, 'w', encoding='utf-8') as f:
        for line in replaced:
            f.write(line)


#act:  {"act_label":"Deny","slot_values":[{"domain":"","slot":"","value":"","pos":""}]}
def validate_act_args(act):
    result = 0
    if act['act_label'] in ['AskHow','Uncertain']:
        return result
    if act['act_label'] in act_labels['noargs_act']: #如果不能有slotvalue
        svs = act['slot_values']
        for sv in svs:
#             if len(sv['domain']) > 0 or len(sv['slot']) > 0 or len(sv['value']) > 0 or len(sv['pos']) > 0:
            if len(sv['slot']) > 0 or len(sv['value']) > 0 or len(sv['pos']) > 0:
                result = 1
                break  
    else:
        svs = act['slot_values']
        for sv in svs:
            if len(sv['domain']) == 0 or len(sv['slot']) ==0 or len(sv['value']) == 0:
                result = 2
    return result

def validate_annoation(file):
    with open(file, 'r',encoding='utf-8') as f:
        data = json.load(f)
    for dialogue in data:
        for round in dialogue['utterances']:
            for act in round['annotation']:
                res = validate_act_args(act) 
                if res >0:
                    print("error: {} conversationId: {} sequenceNo: {} utterance: {}".format(res, dialogue['conversationId'],round['sequenceNo'],round['utterance']))
                
'''
替换影响问题槽位
'''
def check_affected_problem(act):
    svs = act['slot_values']
    for sv in svs:
        if sv['slot'] == '影响问题':
            sv['domain']='问题'
            sv['slot']='症状'

'''
把 标注信息里的问题换成半角问号
'''
def replace_wenhao_in_value(act):
    act_label = act['act_label']
    if act_label == 'AskFor':
        svs = act['slot_values']   
        for sv in svs:
            if sv['value'] == '？':
                sv['value']='?'

'''
查找影响问题槽位
'''
def find_affected_problem(act):
    res = None
    svs = act['slot_values']
    for sv in svs:
        if sv['slot'] == '影响问题':
            res = sv
            break;
    return res
 
def replace_affected_problem_wenhao(input,output):
    with open(input, 'r',encoding='utf-8') as f:
        data = json.load(f)
    for dialogue in data:
        for round in dialogue['utterances']:
            for act in round['annotation']:
                check_affected_problem(act)
                replace_wenhao_in_value(act)
#                 res = find_affected_problem(act) 
#                 if res:
#                     print("+".join(res.values()),'\t',round['utterance'])
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False,indent=4)   



def check_turns(input):
    with open(input, 'r',encoding='utf-8') as f:
        data = json.load(f)
    for dialogue in data:
        turn_num = 0
        for round in dialogue['utterances']:
            valid = round['agentRole'] == 'User' if turn_num % 2 ==0 else  round['agentRole'] == 'Doctor'
            if not valid:
                print('id={} sequenceNo={}'.format(dialogue['conversationId'], turn_num+1))
                continue
            turn_num = turn_num + 1    
                
    
if __name__ == '__main__':
#     replaceFile('annotations_20220508.json','annotations_20220508_1.json')
#     replace_affected_problem_wenhao('annotations_20220508_1.json','annotations_20220508_2.json')
    validate_annoation('annotations_state_20220511.json')
    check_turns('annotations_state_20220511.json')
