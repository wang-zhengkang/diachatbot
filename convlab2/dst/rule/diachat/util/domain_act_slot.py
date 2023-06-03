#encoding=UTF-8
'''
Created on 2022年4月8日

@author: yangjinfeng
'''

'''
这个不需要了，使用convlab2.util.diachat.domain_act_slot
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
        # '影响问题',
        '状态'
      ],
      '饮食': [
        '饮食名',
        '时间',
        '饮食量',
        '成分',
        '成份量',
        # '影响问题',
        '效果'
      ],
      '运动': [
        '运动名',
        '频率',
        '时间',
        '持续时长',
        '强度',
        # '影响问题',
        '效果'
      ],
      '行为': [
        '行为名',
        '频率',
        '时间',
        '持续时长',
        # '影响问题',
        '效果'
      ],
      '治疗': [
        '药品',
        '用药量',
        '用药（治疗）频率',
        '时间',
        '持续时长',
        '适应症',
        '药品类型',
        # '影响问题',
        '治疗名',
        '部位',
        '检查项',
        '检查值',
        '效果'
      ]
    }