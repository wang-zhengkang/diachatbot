#encoding=UTF-8
'''
Created on 2022年5月12日

@author: yangjinfeng
'''
import os
import sys
root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from diachatbot.agent.agent import DiachatAgent
from convlab2.nlu.jointBERT.diachat.nlu import BERTNLU
from convlab2.nlg.sclstm.diachat.sc_lstm import SCLSTM

from convlab2.dst.rule.diachat.dst import RuleDST
from convlab2.policy.mle.diachat.mle import MLE

def demo():
    nlu = BERTNLU()
    dst = RuleDST()
    policy = MLE()
    nlg = SCLSTM()
    diachatbot = DiachatAgent(nlu,dst,policy,nlg)
    while True:
        print("user saying:")
        usr_says = input()
        doctor_answer = diachatbot.response(usr_says)
        print("doctor answering:")
        print("    ",doctor_answer)
        
        print('**********print status***********')
        status = diachatbot.get_current_status()
        status.print_status()
        print('**********print status***********')
        
        if usr_says == '再见':
            break;


# -X utf8
if __name__ == '__main__':
    demo()
        
        
        
        
        
        
        
        
        
        