from convlab2.dst import DST
# from convlab2.dst.rule.diachat.util.state_structure import default_state
from convlab2.util.diachat.state_structure import default_state
from convlab2.util.diachat.state_structure import validate_domain_group
from copy import deepcopy
from collections import Counter
from pprint import pprint
import json
# from convlab2.dst.rule.diachat.util.domain_act_slot import act_labels
from convlab2.util.diachat.domain_act_slot import act_labels
INTENT = {"Inform":"现状","Advice":"建议","AdviceNot":"建议","Explanation":"解释"}
# INTENT = {"Inform":"现状","Advice":"建议","Explanation":"解释"}

class RuleDST(DST):
    """Rule based DST which trivially updates new values from NLU result to states.

    Attributes:
        state(dict):
            Dialog state. Function ``convlab2.util.diachat.state_structure.default_state`` returns a default state.
    """
    def __init__(self):
        super().__init__()
        self.state = default_state()

    def init_session(self, state=None):
        """
        Initialize ``self.state`` with a default state, 
        which ``convlab2.util.diachat.state_structure.default_state`` returns.
        """
        self.state = default_state() if not state else deepcopy(state)


    '''
            当系统接受用户语句并完成NLU之后，根据用户的dialog act 更新状态。
            用户的dialog act结构为[['Inform', '饮食', '饮食名', '莲藕排骨']]
            
            潜在的问题：是否建议没有处理好，下一步要大幅度改这个belief_state的结构
    '''
    def update_belief_state(self, dialog_act=None):
        for intent, domain, slot, value in dialog_act:
            if slot != "" and (slot not in ['是否建议','状态']):
                if intent in ['Inform','Advice','AdviceNot','Explanation']:
                    label = INTENT[intent]
                    if not validate_domain_group(domain,label):
                        continue
                    num = len(self.state['belief_state'][domain][label])
                    if not self.state['belief_state'][domain][label][num - 1][slot]:
                        self.state['belief_state'][domain][label][num - 1][slot] = value
                    else:
                        new_entry = default_state()['belief_state'][domain][label][0]
                        new_entry[slot] = value
                        self.state['belief_state'][domain][label].append(new_entry)
                elif intent == 'AskFor':
                    self.state['askfor_slots'].append([domain, slot, value])
                elif intent == 'AskForSure':
                    self.state['askforsure_slotvs'].append([domain, slot, value])



    def update(self, usr_da = None):
        # 更新 user_action
        self.state['user_action'] = usr_da
        sys_da = self.state['system_action']

        #replaced by yangjinfeng
        domain_array = []
        for da in usr_da:
            if da[1] not in domain_array:
                domain_array.append(da[1])
        self.state['cur_domain'] = "-".join(domain_array)

        # DONE: clean ask slot  ---> AskFor AskForSure
        for domain, slot, value in deepcopy(self.state['askfor_slots']) + deepcopy(self.state['askforsure_slotvs']):
            if [domain, slot] in [x[1:3] for x in sys_da if x[0] in act_labels['doctor_act']]:
                self.state['ask_slots'].remove([domain, slot,value])

        # 更新 belief_state  askfor_slots askforsure_slotvs
        self.update_belief_state(usr_da)
        return self.state


    '''
            当系统完成了Policy的决策后生成响应用户的dialog act, 在进一步调用NLG的同时， 更新自身状态状态。
            主要根据系统的Advice和Explanation更新belief state里的建议和解释
            系统的dialog act结构为[['Adivce', '饮食', '饮食量', '少']]
    '''
    def update_by_sysda(self):
#         self.state['system_action'] = sys_da
        sys_da = self.state['system_action'] #system action在policy部分已传入     
        self.update_belief_state(sys_da)
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
#     pprint(dst.state)
    state_json = json.dumps(dst.state, ensure_ascii=False, sort_keys=True, indent=4)
    print(state_json)

    sys_actions = [["Explanation","问题","时间","空腹"],
                    ["Explanation","问题","时间","餐后两个小时"],
                    ["Advice","行为","频率","一周测两次"],
                    ["AdviceNot","行为","行为名","喝酒"],
                    ["AdviceNot","行为","行为名","吸烟"],
                    ["Explanation","行为","行为名","喝酒"],
                    ["Explanation","问题","症状","血糖升高"]]
    dst.update_by_sysda(sys_actions)
    state_json = json.dumps(dst.state, ensure_ascii=False, sort_keys=True, indent=4)
    print(state_json)







