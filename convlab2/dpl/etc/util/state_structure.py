from convlab2.dpl.etc.util.domain_act_slot import *
import numpy as np
import copy


belief_state_structure = {
    "基本信息": {"现状": []},
    "问题": {"现状": [], "解释": []},
    "饮食": {"现状": [], "解释": [], "建议": []},
    "运动": {"现状": [], "解释": [], "建议": []},
    "行为": {"现状": [], "解释": [], "建议": []},
    "治疗": {"现状": [], "解释": [], "建议": []},
}


def empty_state_entry():
    state_entry = {}
    for d in domain_slot:
        if d not in state_entry:
            state_entry[d] = {}
        for s in domain_slot[d]:
            state_entry[d][s] = ""
    return state_entry


def default_state():
    """
    state:
        sys_da: 上一轮系统动作 
            e.g.: [[a, d, s, v], [a, d, s, v]]
        usr_da: 本轮用户动作
            e.g.: [[a, d, s, v], [a, d, s, v]]
        cur_domain: 本轮用户动作涉及领域
            e.g.: ["问题", "运动"]
        inform_ds: 用户Inform动作涉及的domain, slot
            e.g.: [[d, s], [d, s]]
        askhow_ds: 用户AskHow动作涉及的domain, slot
            e.g.: [[d, s], [d, s]]
        askwhy_ds: 用户AskWhy动作涉及的domain, slot
            e.g.: [[d, s], [d, s]]
        askfor_ds: 用户AskFor动作涉及的domain, slot 系统响应里应该包含Assure
            e.g.: [[d, s], [d, s]]
        askforsure_ds: 用户AskForSure动作涉及的domain, slot 系统响应里应该包含Advice
            e.g.: [[d, s], [d, s]]
        belief_state: 记录用户采取动作后的置信状态 根据用户以及系统动作更新 
        terminate: 结束标志
    """
    state = dict(
        sys_da=[],
        usr_da=[],
        cur_domain=[],
        inform_ds=[],
        askhow_ds=[],
        askwhy_ds=[],
        askfor_ds=[],
        askforsure_ds=[],
        belief_state={},
        terminate=False
    )
    entry = empty_state_entry()
    state["belief_state"] = {
        "基本信息": {"现状": [copy.deepcopy(entry["基本信息"])]},
        "问题": {
            # 注意 这里是数组 在填充状态的时候 在同一个act里的slot-value填充到一个entry 如果不是 则新添加一个entry
            "现状": [copy.deepcopy(entry["问题"])],
            "解释": [copy.deepcopy(entry["问题"])],
        },
        "饮食": {
            "现状": [copy.deepcopy(entry["饮食"])],
            "解释": [copy.deepcopy(entry["饮食"])],
            "建议": [copy.deepcopy(entry["饮食"])],
        },
        "运动": {
            "现状": [copy.deepcopy(entry["运动"])],
            "解释": [copy.deepcopy(entry["运动"])],
            "建议": [copy.deepcopy(entry["运动"])],
        },
        "行为": {
            "现状": [copy.deepcopy(entry["行为"])],
            "解释": [copy.deepcopy(entry["行为"])],
            "建议": [copy.deepcopy(entry["行为"])],
        },
        "治疗": {
            "现状": [copy.deepcopy(entry["治疗"])],
            "解释": [copy.deepcopy(entry["治疗"])],
            "建议": [copy.deepcopy(entry["治疗"])],
        },
    }
    return state


def validate_domain_group(domain, group):
    flag = False
    if domain in belief_state_structure:
        flag = group in belief_state_structure[domain]
    return flag


def domain_group_vectorize(belief_state, domain, group):
    domain_slots2id = dict()
    id2domain_slots = dict()
    for d, slots in domain_slot.items():
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
                    if v != "" and (s not in ["是否建议", "状态"]):  # 临时处理一下这两个
                        slots_with_value.append(s)
            slots_with_value = list(set(slots_with_value))
            for slot in slots_with_value:
                has_slot_vec[domain_slots2id[domain][slot]] = 1
    return has_slot_vec


def belief_state_vectorize(belief_state):
    belief_state_vector = []
    for domain in domain_slot.keys():
        if domain == '基本信息':
            has_slot_vec = domain_group_vectorize(belief_state, domain, '现状')
            belief_state_vector = np.hstack(
                (belief_state_vector, has_slot_vec))
        else:
            has_slot_vec = domain_group_vectorize(belief_state, domain, '现状')
            belief_state_vector = np.hstack(
                (belief_state_vector, has_slot_vec))

            has_slot_vec = domain_group_vectorize(belief_state, domain, '解释')
            belief_state_vector = np.hstack(
                (belief_state_vector, has_slot_vec))

            if domain != '问题':
                has_slot_vec = domain_group_vectorize(
                    belief_state, domain, '建议')
                belief_state_vector = np.hstack(
                    (belief_state_vector, has_slot_vec))

    return belief_state_vector
