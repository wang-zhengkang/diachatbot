#encoding=UTF-8
'''
Created on 2022年5月11日

@author: yangjinfeng
'''
import pprint

def anntAction_to_actionArry(annotation):
    '''
            标注的action转换为扁平结构
    '''
    actionArry = []
    for act in annotation:
        intent = act['act_label']
        for dsv in act['slot_values']:
            quad = []
            quad.append(intent)
            quad.append(dsv['domain'])
            quad.append(dsv['slot'])
            quad.append(dsv['value'])
            actionArry.append(quad)
    return actionArry


def recmmdAction_to_anntAction(recmmdAction):
    '''
        recommends: 
        [
            [['Advice', '行为', '治疗名', '注射胰岛素'], ['Advice', '行为', '持续时长', '终身']], 
            [['Advice', '问题', '治疗名', '注射胰岛素'], ['Advice', '问题', '持续时长', '终身']], 
            [['GeneralExplanation', '问题', '', '']]
        ]
        annotation action:
        [
            {
            'act_label': 'Advice', 
            'slot_values': [{'domain': '行为', 'slot': '治疗名', 'value': '注射胰岛素'}, {'domain': '行为', 'slot': '持续时长', 'value': '终身'}]
            }, 
            {'act_label': 'Advice', 
            'slot_values': [{'domain': '问题', 'slot': '治疗名', 'value': '注射胰岛素'}, {'domain': '问题', 'slot': '持续时长', 'value': '终身'}]
            }, 
            {'act_label': 'GeneralExplanation',
             'slot_values': [{'domain': '问题', 'slot': '', 'value': ''}]
            }
        ]
    '''
    anntActions = []
    for actarry in recmmdAction:
        anntAction = {}
        anntAction["act_label"] = actarry[0][0]
        anntAction["slot_values"] = []
        for act in actarry:
            dsv = {"domain":act[1],"slot":act[2],"value":act[3]}
            anntAction["slot_values"].append(dsv)
        anntActions.append(anntAction)
    return anntActions

if __name__ == '__main__':
    recmmd = [[['Advice', '行为', '治疗名', '注射胰岛素'], ['Advice', '行为', '持续时长', '终身']], [['Advice', '问题', '治疗名', '注射胰岛素'], ['Advice', '问题', '持续时长', '终身']], [['GeneralExplanation', '问题', '', '']]]
    x = recmmdAction_to_anntAction(recmmd)
    print(x)
