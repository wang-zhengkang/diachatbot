#encoding=UTF-8
'''
Created on 2022年6月20日

@author: yangjinfeng
'''
class DiachatSatus:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.input_utterance=None
        self.input_action = None
        self.policy_action = None
        self.output_action  = None # 也就是infered_action
        self.output_utterance  = None
    
    def print_status(self):
        print("input utterance: ",self.input_utterance)
        print("input action: ",self.input_action)
        print("policy action: ",self.policy_action)
        print("output action: ",self.output_action)
        print("output utterance: ",self.output_utterance)

            
        
# turn_status = DiachatSatus()
# 
# def reset():
#     turn_status.reset()
#     
# def set_input_utterance(input_utterance):
#     turn_status.input_utterance = input_utterance
# 
# def set_input_action(input_action):
#     turn_status.input_action = input_action
# 
# def set_policy_action(policy_action):
#     turn_status.policy_action = policy_action
# 
# def set_output_action(output_action):
#     turn_status.output_action = output_action
#     
# def set_output_utterance(output_utterance):
#     turn_status.output_utterance = output_utterance
# 
# def print_status():
#     print("input utterance: ",turn_status.input_utterance)
#     print("input action: ",turn_status.input_action)
#     print("policy action: ",turn_status.policy_action)
#     print("output action: ",turn_status.output_action)
#     print("output utterance: ",turn_status.output_utterance)


# if __name__ == '__main__':
#     reset()
#     set_input_utterance('qqqq')
#     print_status()