all_act = ['Inform', 'Advice', 'AdviceNot', 'Accept', 'Explanation', 'AskForSure', 'AskWhy',
              'AskFor', 'AskHow', 'Assure', 'Deny', 'Uncertain', 'GeneralAdvice', 'Chitchat',
              'GeneralExplanation']

doctor_act = ["Advice", "AdviceNot", "AskFor", "AskForSure", "AskHow", "Assure", "Chitchat",
               "Deny", "Explanation", "GeneralExplanation", "GeneralAdvice"]

usr_act = ["Accept", "AskFor", "AskForSure", "AskHow", "AskWhy", "Assure", "Chitchat", "Inform",
             "Uncertain"]

all_domain = ['基本信息', '行为', '治疗', '问题', '运动', '饮食']

domain_slot = {
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
