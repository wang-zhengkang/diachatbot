#encoding=UTF-8
'''
Created on 2022年5月13日

@author: yangjinfeng
'''
from diachatbot.agent.agent import DiachatAgent2
from convlab2.nlg.sclstm.diachat.sc_lstm import SCLSTM

from convlab2.dst.trade.diachat.trade import DiachatTRADE
from convlab2.policy.mle.diachat.mle import MLE



def demo():
    '''
    demo for DST+Policy+NLG
    '''
    # dst = DiachatTRADE()
    dst=DiachatTRADE('model/TRADE-multiwozdst/HDD100BSZ4DR0.2ACC-0.0435')
    policy = MLE()
    nlg = SCLSTM()
    diachatbot = DiachatAgent2(dst,policy,nlg)
    while True:
        print("user saying:")
        usr_says = input()
        doctor_answer = diachatbot.response(usr_says)
        print("doctor answering:")
        print("    ",doctor_answer)
        if usr_says == '再见':
            break;






if __name__ == '__main__':
    demo()