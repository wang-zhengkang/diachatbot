from convlab2.policy.mle.crosswoz.mle import MLE
from convlab2.dst.rule.crosswoz.dst import RuleDST
from convlab2.util.crosswoz.state import default_state
from convlab2.policy.rule.crosswoz.rule_simulator import Simulator
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.util.crosswoz.lexicalize import delexicalize_da
from convlab2.nlu.jointBERT.crosswoz.nlu import BERTNLU
from convlab2.nlg.template.crosswoz.nlg import TemplateNLG
from convlab2.nlg.sclstm.crosswoz.sc_lstm import SCLSTM
import os
import zipfile
import json
from copy import deepcopy
import random
import numpy as np
from pprint import pprint
import torch


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        for quad in predicts:
            if quad in labels:
                TP += 1
            else:
                FP += 1
        for quad in labels:
            if quad not in predicts:
                FN += 1
    print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP) if (TP + FP) else 0.
    recall = 1.0 * TP / (TP + FN) if (TP + FN) else 0.
    F1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.
    return precision, recall, F1


def evaluate_corpus_f1(policy, data, goal_type=None):
    dst = RuleDST()#包括database和state两部分，初始化的时候user_action和system_action都是空的
    da_predict_golden = []
    delex_da_predict_golden = []
    for task_id, sess in data.items():#data由一个数量为2303的dict组成，具体可参考crosswoz数据介绍
        if goal_type and sess['type']!=goal_type:#校验一下type是否一致，如果不一致，直接跳过
            continue
        dst.init_session()#初始化dst的state参数
        for i, turn in enumerate(sess['messages']):#messages：用户和系统的聊天记录及进一步标注信息
            if turn['role'] == 'usr':#如果是用户说的那一句
                dst.update(usr_da=turn['dialog_act'])#更新'cur_domain'、'request_slots'之类的，要调用dst的代码
                dst.state['user_action'] = turn['dialog_act']#！！！自加代码，否则predict的时候da始终为0
                if i + 2 == len(sess):#如果是倒数第二句话
                    dst.state['terminated'] = True
            else:#如果是系统说的
                for domain, svs in turn['sys_state'].items():#domain有五个，svs是每个领域的槽位和槽值
                    for slot, value in svs.items():
                        if slot != 'selectedResults':
                            dst.state['belief_state'][domain][slot] = value#更新一下dst['belief_state']的状态，#把值粘过去，注意，这里的belief_state是指机器对人那句话的理解值，如'你好，可以帮我安排一个人均消费50-100元，能吃到香椿炒鸡蛋的餐馆吗？'就是={'名称':'','推荐菜': '香椿炒鸡蛋', '人均消费': '50-100元', '评分': '', '周边景点': '', '周边餐馆': '', '周边酒店': ''}
                            #dst.state['system_action'] = turn['dialog_act']  # ！！！这行代码不应该增加了，因为sys_act作为输入端时理应全部为0
                golden_da = turn['dialog_act']#注意这里是在else：后面的，所以是系统一次回答的['dialog_act']，举个栗子，[['Recommend', '餐馆', '名称', '圣霖荷园'], ['Recommend', '餐馆', '名称', '将进酒客栈']]

                predict_da = policy.predict(deepcopy(dst.state))#这是预测的da（dialog_act），policy模块预测的，具体在mle.py代码里面
                # print(golden_da)
                # print(predict_da)
                # print()
                # if 'Select' in [x[0] for x in sess['messages'][i - 1]['dialog_act']]:
                da_predict_golden.append({
                    'predict': predict_da,#形如[['Inform', '餐馆', '周边景点', '黄花城水长城']]
                    'golden': golden_da#形如[['Inform', '餐馆', '周边景点', '延寿寺'], ['Inform', '餐馆', '周边景点', '鳞龙山'], ['Inform', '餐馆', '周边景点', '黄花城水长城']]
                })
                delex_da_predict_golden.append({
                    'predict': delexicalize_da(predict_da),#形如['Inform+餐馆+周边景点+1']
                    'golden': delexicalize_da(golden_da)#形如['Inform+餐馆+周边景点+1', 'Inform+餐馆+周边景点+2', 'Inform+餐馆+周边景点+3']
                })
                # print(delex_da_predict_golden[-1])
                dst.state['system_action'] = golden_da
        # break
    print('origin precision/recall/f1:', calculateF1(da_predict_golden))#严格环境下的F1
    print('delex precision/recall/f1:', calculateF1(delex_da_predict_golden))#没那么严格环境下的F1，即去词化(delexicalize)


def end2end_evaluate_simulation(policy):
    nlu = BERTNLU()
    nlg_usr = TemplateNLG(is_user=True, mode='auto_manual')
    nlg_sys = TemplateNLG(is_user=False, mode='auto_manual')
    # nlg_usr = SCLSTM(is_user=True, use_cuda=False)
    # nlg_sys = SCLSTM(is_user=False, use_cuda=False)
    usr_policy = Simulator()
    usr_agent = PipelineAgent(nlu, None, usr_policy, nlg_usr, name='user')
    sys_policy = policy
    sys_dst = RuleDST()
    sys_agent = PipelineAgent(nlu, sys_dst, sys_policy, nlg_sys, name='sys')
    sess = BiSession(sys_agent=sys_agent, user_agent=usr_agent)

    task_finish = {'All': list(), '单领域': list(), '独立多领域': list(), '独立多领域+交通': list(), '不独立多领域': list(),
                    '不独立多领域+交通': list()}
    simulate_sess_num = 100
    repeat = 10
    random_seed = 2019
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random_seeds = [random.randint(1, 2**32-1) for _ in range(simulate_sess_num * repeat * 10000)]
    while True:
        sys_response = ''
        random_seed = random_seeds[0]
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random_seeds.pop(0)
        sess.init_session()
        # print(usr_policy.goal_type)
        if len(task_finish[usr_policy.goal_type]) == simulate_sess_num*repeat:
            continue
        for i in range(15):
            sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
            # print('user:', user_response)
            # print('sys:', sys_response)
            # print(session_over, reward)
            # print()
            if session_over is True:
                task_finish['All'].append(1)
                task_finish[usr_policy.goal_type].append(1)
                break
        else:
            task_finish['All'].append(0)
            task_finish[usr_policy.goal_type].append(0)
        print([len(x) for x in task_finish.values()])
        # print(min([len(x) for x in task_finish.values()]))
        if len(task_finish['All']) % 100 == 0:
            for k, v in task_finish.items():
                print(k)
                all_samples = []
                for i in range(repeat):
                    samples = v[i * simulate_sess_num:(i + 1) * simulate_sess_num]
                    all_samples += samples
                    print(sum(samples), len(samples), (sum(samples) / len(samples)) if len(samples) else 0)
                print('avg', (sum(all_samples) / len(all_samples)) if len(all_samples) else 0)
        if min([len(x) for x in task_finish.values()]) == simulate_sess_num*repeat:
            break
        # pprint(usr_policy.original_goal)
        # pprint(task_finish)
    print('task_finish')
    for k, v in task_finish.items():
        print(k)
        all_samples = []
        for i in range(repeat):
            samples = v[i * simulate_sess_num:(i + 1) * simulate_sess_num]
            all_samples += samples
            print(sum(samples), len(samples), (sum(samples) / len(samples)) if len(samples) else 0)
        print('avg', (sum(all_samples) / len(all_samples)) if len(all_samples) else 0)


def da_evaluate_simulation(policy):
    usr_policy = Simulator()
    usr_agent = PipelineAgent(None, None, usr_policy, None, name='user')
    sys_policy = policy
    sys_dst = RuleDST()
    sys_agent = PipelineAgent(None, sys_dst, sys_policy, None, name='sys')
    sess = BiSession(sys_agent=sys_agent, user_agent=usr_agent)

    task_finish = {'All': list(), '单领域': list(), '独立多领域': list(), '独立多领域+交通': list(), '不独立多领域': list(),
                    '不独立多领域+交通': list()}
    simulate_sess_num = 100
    repeat = 10
    random_seed = 2019
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random_seeds = [random.randint(1, 2**32-1) for _ in range(simulate_sess_num * repeat * 10000)]
    while True:
        sys_response = []
        random_seed = random_seeds[0]
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random_seeds.pop(0)
        sess.init_session()
        # print(usr_policy.goal_type)
        if len(task_finish[usr_policy.goal_type]) == simulate_sess_num*repeat:
            continue
        for i in range(15):
            sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
            # print('user:', user_response)
            # print('sys:', sys_response)
            # print(session_over, reward)
            # print()
            if session_over is True:
                # pprint(sys_agent.tracker.state)
                task_finish['All'].append(1)
                task_finish[usr_policy.goal_type].append(1)
                break
        else:
            task_finish['All'].append(0)
            task_finish[usr_policy.goal_type].append(0)
        print([len(x) for x in task_finish.values()])
        # print(min([len(x) for x in task_finish.values()]))
        if len(task_finish['All']) % 100 == 0:
            for k, v in task_finish.items():
                print(k)
                all_samples = []
                for i in range(repeat):
                    samples = v[i * simulate_sess_num:(i + 1) * simulate_sess_num]
                    all_samples += samples
                    print(sum(samples), len(samples), (sum(samples) / len(samples)) if len(samples) else 0)
                print('avg', (sum(all_samples) / len(all_samples)) if len(all_samples) else 0)
        if min([len(x) for x in task_finish.values()]) == simulate_sess_num*repeat:
            break
        # pprint(usr_policy.original_goal)
        # pprint(task_finish)
    print('task_finish')
    for k, v in task_finish.items():
        print(k)
        all_samples = []
        for i in range(repeat):
            samples = v[i * simulate_sess_num:(i + 1) * simulate_sess_num]
            all_samples += samples
            print(sum(samples), len(samples), (sum(samples) / len(samples)) if len(samples) else 0)
        print('avg', (sum(all_samples) / len(all_samples)) if len(all_samples) else 0)


if __name__ == '__main__':
    random_seed = 2019
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    test_data = os.path.abspath(os.path.join(os.path.abspath(__file__),'../../../../../data/crosswoz/test.json.zip'))
    test_data = read_zipped_json(test_data, 'test.json')
    policy = MLE()
    for goal_type in ['单领域','独立多领域','独立多领域+交通','不独立多领域','不独立多领域+交通',None]:
        print(goal_type)
        evaluate_corpus_f1(policy, test_data, goal_type=goal_type)
    #da_evaluate_simulation(policy)#跑一遍
    #end2end_evaluate_simulation(policy)
