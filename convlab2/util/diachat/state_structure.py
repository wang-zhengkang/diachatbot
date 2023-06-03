# encoding=utf-8
# import pprint
import json
from convlab2.util.diachat.domain_act_slot import *
import numpy as np
import copy

belief_state_structure = {
    "基本信息": {
        "现状": []
    },
    "问题": {
        "现状": [],
        "解释": []
    },
    "饮食": {
        "现状": [],
        "解释": [],
        "建议": []
    },
    "运动": {
        "现状": [],
        "解释": [],
        "建议": []
    },
    "行为": {
        "现状": [],
        "解释": [],
        "建议": []
    },
    "治疗": {
        "现状": [],
        "解释": [],
        "建议": []
    }
}


def empty_state_entry():
    state_entry = {}
    for d in domain_slots:
        if d not in state_entry:
            state_entry[d] = {}
        for s in domain_slots[d]:
            #             if s == '既往史':
            #                 state_entry[d][s]=[]
            #             else:
            state_entry[d][s] = ''
    return state_entry


def default_state():
    state = dict(user_action=[],  # 本轮的user dialog act
                 system_action=[],  # 上一轮的sys dialog act
                 belief_state={},
                 cur_domain=None,
                 # 类似于原来的request_slots，对应于AskFor,数组里的元素是[domain,slot,value]， 最后的一个或多个是[domain,slot,'?']
                 # 系统响应里应该包含advice
                 askfor_slots=[],
                 # 对应于AskForSure,数组里的元素是[domain,slot,value],系统响应里应该包含assure
                 askforsure_slotvs=[],
                 terminated=False,
                 history=[])
    entry = empty_state_entry()
    state['belief_state'] = {
        '基本信息': {
            '现状': [copy.deepcopy(entry['基本信息'])]
        },
        '问题': {
            '现状': [copy.deepcopy(entry['问题'])],  # 注意，这里是数组，在填充状态的时候，在同一个act里的slot-value填充到一个entry，如果不是，则新添加一个entry
            '解释': [copy.deepcopy(entry['问题'])],
        },
        '饮食': {
            '现状': [copy.deepcopy(entry['饮食'])],
            '解释': [copy.deepcopy(entry['饮食'])],
            '建议': [copy.deepcopy(entry['饮食'])],
        },
        '运动': {
            '现状': [copy.deepcopy(entry['运动'])],
            '解释': [copy.deepcopy(entry['运动'])],
            '建议': [copy.deepcopy(entry['运动'])],
        },
        '行为': {
            '现状': [copy.deepcopy(entry['行为'])],
            '解释': [copy.deepcopy(entry['行为'])],
            '建议': [copy.deepcopy(entry['行为'])],
        },
        '治疗': {
            '现状': [copy.deepcopy(entry['治疗'])],
            '解释': [copy.deepcopy(entry['治疗'])],
            '建议': [copy.deepcopy(entry['治疗'])],
        },

    }
    return state


def validate_domain_group(domain, group):
    ok = False
    if domain in belief_state_structure:
        ok = group in belief_state_structure[domain]
    return ok


def domain_group_vectorize(belief_state, domain, group):
    has_slot_vec = np.zeros(len(domain_slots2id[domain]))
    if domain in belief_state:
        domain_belief = belief_state[domain]
        if group in domain_belief:
            slot_values_group = domain_belief[group]
            slots_with_value = []
            for svs in slot_values_group:
                for s, v in svs.items():
                    if v != '' and (s not in ['是否建议', '状态']):  # 临时处理一下这两个
                        slots_with_value.append(s)
            slots_with_value = list(set(slots_with_value))
            for slot in slots_with_value:
                has_slot_vec[domain_slots2id[domain][slot]] = 1
    return has_slot_vec


# belief_state = state['belief_state']
def belief_state_vectorize(belief_state):
    belief_state_vector = []
    for domain in domain_slots.keys():
        if domain == '基本信息':
            has_slot_vec = domain_group_vectorize(belief_state, domain, '现状')
            belief_state_vector = np.hstack((belief_state_vector, has_slot_vec))
        else:
            has_slot_vec = domain_group_vectorize(belief_state, domain, '现状')
            belief_state_vector = np.hstack((belief_state_vector, has_slot_vec))

            has_slot_vec = domain_group_vectorize(belief_state, domain, '解释')
            belief_state_vector = np.hstack((belief_state_vector, has_slot_vec))

            if domain != '问题':
                has_slot_vec = domain_group_vectorize(belief_state, domain, '建议')
                belief_state_vector = np.hstack((belief_state_vector, has_slot_vec))

    return belief_state_vector


if __name__ == '__main__':
    # print(json.dumps(default_state()['belief_state'],indent=2,ensure_ascii=False))
    # print(json.dumps(default_state(),indent=2,ensure_ascii=False))
    # print(json.dumps(empty_state_entry(),indent=2,ensure_ascii=False))
    print(validate_domain_group('基本信息', '现状'))
    print(validate_domain_group('问题', '建议'))
    print(validate_domain_group('行为', '现状'))
