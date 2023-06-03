#encoding=utf-8
# import pprint
import json
from convlab2.dst.rule.diachat.util.domain_act_slot import *
import copy

'''
这个不需要了，使用convlab2.util.diachat.state_structure
'''
def empty_state_entry():
    state_entry = {}
    for d in domain_slots:
        if d not in state_entry:
            state_entry[d]={}
        for s in domain_slots[d]:
            if s == '既往史':
                state_entry[d][s]=[]
            else:
                state_entry[d][s]=''
    return state_entry


def default_state():
    state = dict(user_action=[], 
                 system_action=[], 
                 belief_state={}, 
                 cur_domain=None, 
                 askfor_slots=[],
                 askforsure_slots=[],
                 terminated=False,
                 history=[])
    entry = empty_state_entry()
    state['belief_state'] = {
        '基本信息':{
            '现状': [copy.deepcopy(entry['基本信息'])]
            },
        '问题': {
            '现状':[copy.deepcopy(entry['问题'])], #注意，这里是数组，在填充状态的时候，在同一个act里的slot-value填充到一个entry，如果不是，则新添加一个entry
            '解释':[copy.deepcopy(entry['问题'])],
            '建议':[copy.deepcopy(entry['问题'])]
            },
        '饮食': {
            '现状':[copy.deepcopy(entry['饮食'])],
            '解释':[copy.deepcopy(entry['饮食'])],
            '建议':[copy.deepcopy(entry['饮食'])],    
            },
        '运动': {
            '现状':[copy.deepcopy(entry['运动'])],
            '解释':[copy.deepcopy(entry['运动'])],
            '建议':[copy.deepcopy(entry['运动'])],    
            },
        '行为': {
            '现状':[copy.deepcopy(entry['行为'])],
            '解释':[copy.deepcopy(entry['行为'])],
            '建议':[copy.deepcopy(entry['行为'])],    
            },
        '治疗': {
            '现状':[copy.deepcopy(entry['治疗'])],
            '解释':[copy.deepcopy(entry['治疗'])],
            '建议':[copy.deepcopy(entry['治疗'])],    
            },
    
        }
    return state




if __name__ == '__main__':
    print(json.dumps(default_state()['belief_state'],indent=2,ensure_ascii=False))
