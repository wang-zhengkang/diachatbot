from convlab2.policy.gdpl.diachat.domain_act_slot import *
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
            state_entry[d][s] = ''
    return state_entry


def default_state():
    state = dict(usr_action=[],  # 本轮user dialog act
                 sys_action=[],  # 上一轮sys dialog act
                 belief_state={},
                 cur_domain=None,
                 # askfor_slots类似于原来的request_slots 对应于AskFor
                 # 数组里的元素是[domain,slot,value] 最后的一个或多个是[domain,slot,'?']
                 # 系统响应里应该包含Advice
                 askfor_slots=[],
                 # askforsure_slotvs对应于AskForSure
                 # 数组里的元素是[domain,slot,value],系统响应里应该包含Assure
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
    domain_slots2id = dict()
    id2domain_slots = dict()
    for d, slots in domain_slots.items():
        domain_slots2id[d] = dict((slots[i], i) for i in range(len(slots)))
        id2domain_slots[d] = dict((i, slots[i]) for i in range(len(slots)))
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
