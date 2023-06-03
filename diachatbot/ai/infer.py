#encoding=UTF-8
'''
Created on 2022年4月15日

@author: yangjinfeng
'''

'''
+─────────────────────────+─────────────────────+─────────────────────+──────────────────────────────+
| three major categories  | recommand_type      | recommend_method    | description                  |
+─────────────────────────+─────────────────────+─────────────────────+──────────────────────────────+
|                         | Advice or AdviceNot | advice              |advice sth or advice not sth  |
| three specific answers  | Assure or Deny      | assure              |assure sth or deny sth        |
|_________________________| Explanation         | explanation         |explanation by sth            |
|                         | AskFor              | ask_for             |ask for sth                   |
| three specific questions| AskForSure          | ask_for_sure        |to make sth sure              |
|_________________________| AskHow              | ask_how             |ask about how sth is going    |
|                         | GeneralAdvice       | general_advice      |give some general advice      |
| three general responses | GeneralExplanation  | general_explanation |give some general explanation |
|                         | Chitchat            | chitchat            |including greetings, thanks   |    
+─────────────────────────+─────────────────────+─────────────────────+──────────────────────────────+
    
'''
import json
from diachatbot.ai.ckan.CKAN_robot_interface import predict
from diachatbot.ai.ckan.CKAN_tokenization import user_id_sym_dis_id
class Inference(object):


    '''
    delexicalized actions : [['Advice','治疗','治疗名'],['Advice','治疗','持续时长'],[GeneralAdvice,'','','']]
    '''
    
    def recommend_by_delex_da(self, state, delex_das):
        '''
                    根据delex_das解析出act label和domain、slot，推荐的时候可以不用考虑slot，或者把slot作为推荐的参考
                    推荐的时候给出推荐的slot-value，最终和act label、domain组成完整的dialog acts
        
        Args:
            state (dict): 系统当前的状态, state的格式和样例已具备
            delex_das (list) :去词汇化的dialog acts，由多分类模型给出 , 表示系统的回复意图、回复的领域和回复内容的要素
                                                            
        Returns:
            slot-values (list of list or dict): 返回完整的dialog acts
        
                    举例：
                    比如，delex_das = [['Advice','治疗','治疗名'],['Advice','治疗','持续时长']]
                    表示系统将要给出治疗领域治疗名称和持续时长两方面的建议。
                    如果 建议的治疗名和持续时长分别是注射胰岛素和终身，那么返回值就是
                    [[Advice','治疗','治疗名','注射胰岛素'],['Advice','治疗','持续时长','终身']]

        ''' 
        das_dict={}  #key是 'Advice-治疗'，value是 ['治疗名','持续时长']
        for delx_da in delex_das:
            key = delx_da[0]+'-'+delx_da[1]
            if key not in das_dict:
                das_dict[key] = [delx_da[2]]
            else:
                das_dict[key].append(delx_da[2])
        
        recommend_das = []
        for actdomain in das_dict:
            act_domain = actdomain.split('-')
            act = act_domain[0]
            domain = act_domain[1]
            slots = das_dict[actdomain]
            slotvalues = self.recommend(state, act, domain, slots)
            act_groups = self.get_da(act, domain, slotvalues)
            recommend_das = recommend_das + act_groups
            
        return recommend_das
    
    
    
    
    def recommend(self,state, recommend_type, domain, slots=None):
        '''
        Args:
            state (dict): 系统当前的状态,state (dict): 系统当前的状态, state的格式和样例已具备，可以只考虑belief_state
            recommend_type (str) :对应系统端的dialog act label                
                Advice: 建议，希望给出建议什么或者不建议什么
                Assure: 肯定，当用户的act是AskForSure的时候,系统给出肯定或者否定，
                Explanation: 解释，希望给出具体的解释
                AskFor: 系统给出要追问的问题
                AskForSure: 系统希望用户确认可能的现状
                AskHow: 笼统的AskFor
                GeneralAdvice: 泛泛的建议
                GeneralExplanation: 泛泛的解释
                Chitchat:  闲聊
            domain (str): 推荐的领域，六个领域：问题、饮食、行为、运动、治疗、基本信息
            slots (list): 可参考的slots，系统可以推荐给定slots的值，也可以是当前domain的其他slots的值
        Returns:
            slot-values (list of list or dict):

        '''
        res = [];#slot-values,结构是[{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n}]
        #three specific answers 系统做出有针对性地回答
        if recommend_type == 'Advice' or recommend_type == 'AdviceNot':
            res = self.advice(state,domain,slots)
        elif  recommend_type == 'Explanation':
            res = self.explanation(state,domain,slots)
        elif recommend_type == 'Assure' or recommend_type == 'Deny':
            res = self.assure(state,domain,slots)
        #three specific questions 系统为了更好的了解用户情况，主动询问用户的具体情况
        elif recommend_type == 'AskFor':
            res = self.ask_for(state,domain,slots)
        elif recommend_type == 'AskForSure':
            res = self.ask_for_sure(state,domain,slots)
        elif recommend_type == 'AskHow':
            res = self.ask_how(state,domain,slots)
        #three general responses 系统给出泛泛的回应，不考虑用户的当前情况。   z这三类推荐可什么都不做，下一步直接交给NLG部分生成回复。
        elif recommend_type in ['GeneralAdvice','GeneralExplanation','Chitchat']:
            res = []
                
        return res
    
    #slotvalues :  [{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n}]
    def get_da(self, act_label, domain, slotvalues):
        act_groups = []
        if len(slotvalues) == 0 and act_label in ['GeneralAdvice','GeneralExplanation','Chitchat']:
            act_groups.append([[act_label,domain,'','']])
        print(act_label, domain, slotvalues)
        #if len(slotvalues) == 1:
            #slotvalues = [slotvalues]
        for svs in slotvalues:
            act_group = []
            for slot in svs:
                act = [act_label, domain,slot,svs[slot]]
                act_group.append(act)
            act_groups.append(act_group)
        
        return act_groups
    
    
    
    def advice(self,state,domain,slots=None):
        '''
                       建议什么或者不建议什么，可以作为对用户askfor的直接回应，如果是回应用户的askfor，参数slot是就是state里的askfor_slots     
                        返回 [{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n, flag:False}]
            flag:False表示不建议
        '''
        if domain == '饮食':
            user_id_sym_dis_id(state)#根据不同的state生成不同的predict_patient_disease_symptom.npy文件
            recommend_food_id, recommend_food_name, recommend_probability=predict()#加载文件，预测当前state下的回答
            res = []
            for food_name in recommend_food_name[0:3]:
                res.append({slots[0]:food_name})
        return res
    
    
    
    def assure(self,state,domain,slots=None):
        '''
                    用户的act是AskForSure的时候,系统给出肯定或者否定，用户 AskForSure以及slot-value)记录在state里 askforsure_slotvs=[],
                    返回用户AskForSure携带的slot-value，形如：
        [{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n, flag:False}]
        flag:False表示Deny
        '''
        return []
    
    
    
    def explanation(self,state,domain,slots=None):
        '''
                        希望给出具体的解释，返回值体现了解释的信息要素
                        返回 [{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n}]
        '''
        return []


    
    def ask_for(self,state,domain,slots=None):
        '''
                    系统给出要追问的问题，就是推断出追问的槽位
                    返回[{slot1:value1,slot2:},{slot_m:value_m,slot_n:}]
        
        '''
        return []
    
    
    def ask_for_sure(self,state,domain,slots=None):
        '''
                    系统推断出患者可能的现状，并希望用户予以确认
                    返回[{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n}]
        '''
        return []
    
    
    
    def ask_how(self,state,domain,slots=None):
        '''
                    表示医生问患者的行为、饮食等情况，arguments表示与问题相关的信息，
        slot是有value的，表示具体的行为、治疗、饮食是如何实施的。
                    这类问题可以看成是笼统的ask_for，在口语对话中经常会出现ask_how这样的问句，比如“你晚饭咋吃的呢”，这个问题可以包含“吃了什么”、“吃了多少”、“啥时候吃的”。
                    返回[{slot1:value1,slot2:value2},{slot_m:value_m,slot_n:value_n}]
        '''
        return []
    
    
    '''
                系统给出泛泛的回应，不考虑用户的当前情况。   z这三类推荐可什么都不做，下一步直接交给NLG部分生成回复。
    '''
    def general_advice(self,state,domain):
        
        return []
    
    
    def general_explanation(self,state,domain):
        
        return []
    
    def chitchat(self,state):
        
        return []
        
if __name__ == '__main__':
    #path = '输入.json'
    path = '输入.json'
    f = open(path, 'r', encoding='utf-8')
    policy_output = json.load(f)
    inf = Inference()
    for num in range(0,1):
         str_num = str(num)
         print(str_num,":")
         print(policy_output[str_num]['prediction'])
         kg_action = inf.recommend_by_delex_da(policy_output[str_num]['state'], policy_output[str_num]['prediction'])
         #print("golden_action:",policy_output[str_num]['golden_action'])#这是希望得到的输出
         print("kg_action:", kg_action)#这是实际输出