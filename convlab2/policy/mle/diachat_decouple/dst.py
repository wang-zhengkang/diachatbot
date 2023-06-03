from convlab2.dst import DST
from convlab2.policy.gdpl.diachat.util.state_structure import default_state, validate_domain_group
from copy import deepcopy
from pprint import pprint
import json

INTENT = {"Inform": "现状", "Advice": "建议", "AdviceNot": "建议", "Explanation": "解释"}


class RuleDST(DST):
    """
    Rule based DST update new values from NLU result to states.

    Attributes:
        state(dict): Dialog state. Function 'state_structure.default_state' return a default state.
    """

    def __init__(self):
        super().__init__()
        self.state = default_state()

    def init_session(self, state=None):
        """
        Initialize 'self.state' with a default state, which 'state_structure.default_state' return.
        """
        self.state = default_state() if not state else deepcopy(state)

    """
    当系统接受用户语句并完成NLU之后 根据用户的dialog act(da)更新状态。
    用户的dialog act结构为[['Inform', '饮食', '饮食名', '莲藕排骨']]
    潜在的问题: 是否建议没有处理好 下一步要大幅度改这个belief_state的结构
    """

    def update_belief_state(self, dialog_act=None):

        for act, domain, slot, value in dialog_act:
            if slot != "none" and (slot not in ['是否建议', '状态']):
                if act in ['Inform', 'Advice', 'AdviceNot', 'Explanation']:
                    label = INTENT[act]
                    # 判断belif_state的domain是否包含label
                    if not validate_domain_group(domain, label):
                        continue
                    num = len(self.state['belief_state'][domain][label])
                    # 如果slot为空 说明是为该domain-label-slot第一次添加value
                    if not self.state['belief_state'][domain][label][num - 1][slot]:
                        self.state['belief_state'][domain][label][num - 1][slot] = value
                    else:
                        new_entry = default_state()['belief_state'][domain][label][0]
                        new_entry[slot] = value
                        self.state['belief_state'][domain][label].append(new_entry)
                elif act == 'AskFor':
                    self.state['askfor_dsv'].append([domain, slot, value])
                elif act == 'AskForSure':
                    self.state['askforsure_dsv'].append([domain, slot, value])

    def update(self, usr_da=None):
        # update usr_action
        self.state['usr_action'] = usr_da
        sys_da = self.state['sys_action']

        # update cur_domain
        # update前清空一下
        self.state['cur_domain'] = []
        for da in usr_da:
            if da[1] not in self.state['cur_domain']:
                self.state['cur_domain'].append(da[1])

        # update belief_state askfor_slots askforsure_slotvs
        self.update_belief_state(usr_da)
        return self.state

    '''
    当系统完成了Policy的决策后生成响应用户的dialog act, 在进一步调用NLG的同时更新自身状态。
    主要根据系统的Advice和Explanation更新belief state里的建议和解释
    系统的dialog act结构为[['Adivce', '饮食', '饮食量', '少']]
    '''

    def update_by_sysda(self, sys_action):
        self.state['sys_action'] = sys_action
        # sys action在policy部分已传入?
        # sys_da = self.state['sys_action']
        self.update_belief_state(sys_action)
        return self.state


if __name__ == '__main__':
    dst = RuleDST()
    dst.init_session()
    # 输入的形式：intent,domain,slot,value
    usr_actions = [['Inform', '饮食', '饮食名', '莲藕排骨'],
                   ['Inform', '饮食', '饮食名', '汤'],
                   ['Inform', '问题', '血糖值', '升高'],
                   ['AskForSure', '行为', '行为名', '喝']]
    dst.update(usr_actions)
    pprint(dst.state)

    sys_actions = [["Explanation", "问题", "时间", "空腹"],
                   ["Explanation", "问题", "时间", "餐后两个小时"],
                   ["Advice", "行为", "频率", "一周测两次"],
                   ["AdviceNot", "行为", "行为名", "喝酒"],
                   ["AdviceNot", "行为", "行为名", "吸烟"],
                   ["Explanation", "行为", "行为名", "喝酒"],
                   ["Explanation", "问题", "症状", "血糖升高"]]
    dst.update_by_sysda(sys_actions)
    pprint(dst.state)
