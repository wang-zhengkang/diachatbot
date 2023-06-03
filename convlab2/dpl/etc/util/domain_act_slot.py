act_labels = ['Inform', 'Advice', 'AdviceNot', 'Accept', 'Explanation', 'AskForSure', 'AskWhy',
             'AskFor', 'AskHow', 'Assure', 'Deny', 'Uncertain', 'GeneralAdvice', 'Chitchat',
             'GeneralExplanation']

sys_act = ["Advice", "AdviceNot", "AskFor", "AskForSure", "AskHow", "Assure", "Chitchat",
           "Deny", "Explanation", "GeneralExplanation", "GeneralAdvice"]

usr_act = ["Accept", "AskFor", "AskForSure", "AskHow", "AskWhy", "Assure", "Chitchat", "Inform",
           "Uncertain"]

domains = ['问题', '饮食', '行为', '运动', '治疗', '基本信息']

# 用户目标分类
SORTS = ['current', 'AskForSure', 'AskFor', 'AskHow', 'AskWhy', 'Chitchat']

domain_slot = {
    '问题': [
        '疾病',
        '症状',
        '症状部位',
        '持续时长',
        '血糖值',
        '时间',
        '状态'
    ],
    '饮食': [
        '饮食名',
        '时间',
        '饮食量',
        '成分',
        '成份量',
        '效果'
    ],
    '行为': [
        '行为名',
        '频率',
        '时间',
        '持续时长',
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
    ],

    '基本信息': [
        '身高',
        '体重',
        '年龄',
        '性别',
        '既往史'
    ]
    
    
    
    
}

domain_slot2id = dict()
id2domain_slot = dict()

for domain, slot in domain_slot.items():
    domain_slot2id[domain] = dict((slot[i], i) for i in range(len(slot)))
    id2domain_slot[domain] = dict((i, slot[i]) for i in range(len(slot)))
