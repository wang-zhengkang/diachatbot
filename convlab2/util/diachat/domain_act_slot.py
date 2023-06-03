#encoding=UTF-8
'''
Created on 2022年4月8日

@author: yangjinfeng
'''

'''
所有的act labels
'''
act_labels={
    "noargs_act":['Accept','AskWhy','Assure','Deny','GeneralAdvice','Chitchat','GeneralExplanation'],
    "doctor_act":['Advice','AdviceNot','Explanation','AskForSure','AskFor','AskHow','Assure','Deny','GeneralAdvice','Chitchat','GeneralExplanation'],
    "user_act":['Inform','Accept','AskForSure','AskWhy','AskFor','AskHow','Assure','Deny','Uncertain','Chitchat'],
    "all_act":['Inform','Advice','AdviceNot','Accept','Explanation','AskForSure','AskWhy','AskFor','AskHow','Assure','Deny','Uncertain','GeneralAdvice','Chitchat','GeneralExplanation']
    }


'''
所有的domain和slots
'''
domain_slots={
      '基本信息': [
        '身高',
        '体重',
        '年龄',   
        '性别',
        '既往史'
      ],
      '问题': [
        '疾病',
        '症状',
        '症状部位',
        '持续时长',
        '血糖值',
        '时间',
      ],
      '饮食': [
        '饮食名',
        '时间',
        '饮食量',
        '成分',
        '成份量',
        '效果'
      ],
      '运动': [
        '运动名',
        '频率',
        '时间',
        '持续时长',
        '强度',
        '效果'
      ],
      '行为': [
        '行为名',
        '频率',
        '时间',
        '持续时长',
        '效果'
      ],
      '治疗': [
        '药品',
        '用药量',
        '用药治疗频率',
        '时间',
        '持续时长',
        '适应症',
        '药品类型',
        '治疗名',
        '部位',
        '检查项',
        '检查值',
        '效果'
      ]
    }

domain_slots2id=dict()
id2domain_slots=dict()

for d,slots in domain_slots.items():
    domain_slots2id[d] = dict((slots[i], i) for i in range(len(slots)))
    id2domain_slots[d] = dict((i, slots[i]) for i in range(len(slots)))
    