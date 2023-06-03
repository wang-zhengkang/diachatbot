import os
import json
import zipfile
from convlab2.policy.gdpl.diachat.dst import RuleDST
from convlab2.policy.gdpl.diachat.gdpl import GDPL
from copy import deepcopy


def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        labels = item[0]
        predicts = item[1]
        for quad in predicts:
            if quad in labels:
                TP += 1
            else:
                FP += 1
        for quad in labels:
            if quad not in predicts:
                FN += 1
    print(f"TP:{TP}")
    print(f"FP:{FP}")
    print(f"FN:{FN}")
    precision = 1.0 * TP / (TP + FP) if (TP + FP) else 0.
    recall = 1.0 * TP / (TP + FN) if (TP + FN) else 0.
    F1 = 2.0 * precision * recall / \
        (precision + recall) if (precision + recall) else 0.
    return precision, recall, F1


def anntAction_to_actionArry(annotation):
    '''
    标注的action转换为扁平结构
    '''
    actionArry = []
    for act in annotation:
        intent = act['act_label']
        for dsv in act['slot_values']:
            quad = []
            quad.append(intent)
            quad.append(dsv['domain'])
            quad.append(dsv['slot'])
            quad.append(dsv['value'])
            actionArry.append(quad)
    return actionArry

# askfor_askforsure == AskFor or AskForSure


def annt_to_askfor_slots(annotation, askfor_askforsure):
    slotsArry = []
    if askfor_askforsure not in ['AskFor', 'AskForSure']:
        return slotsArry
    for act in annotation:
        intent = act['act_label']
        if intent == askfor_askforsure:
            for dsv in act['slot_values']:
                quad = []
                quad.append(dsv['domain'])
                quad.append(dsv['slot'])
                quad.append(dsv['value'])
                slotsArry.append(quad)
    return slotsArry


def build_test_state():
    part = 'test'
    raw_data = {}
    data = {}
    data[part] = []
    archive = zipfile.ZipFile('data/diachat/test.json.zip', 'r')
    with archive.open('{}.json'.format(part), 'r') as f:
        raw_data[part] = json.load(f)

    dst = RuleDST()
    for conversation in raw_data[part]:
        sess = conversation['utterances']
        dst.init_session()

        for i, turn in enumerate(sess):

            domains = turn['domain']

            if domains == '':
                domains = 'none'

            sys_action = []  # 系统的action，作为训练数据的Y

            if turn['agentRole'] == 'User':
                #                         dst.state['user_domain'] = domains
                dst.state['cur_domain'] = domains

                user_action = anntAction_to_actionArry(turn['annotation'])
                askfor_slots = annt_to_askfor_slots(
                    turn['annotation'], 'AskFor')
                askforsure_slotvs = annt_to_askfor_slots(
                    turn['annotation'], 'AskForSure')
                dst.state['askfor_slots'] = askfor_slots
                dst.state['askforsure_slotvs'] = askforsure_slotvs
                dst.state['user_action'] = user_action

                if i + 2 == len(sess):
                    dst.state['terminated'] = True

            else:  # agentRole=Doctor
                dst.state['belief_state'] = turn['sys_state_init']
                sys_action = anntAction_to_actionArry(turn['annotation'])

                test_X = deepcopy(dst.state)  # 553
                test_Y = deepcopy(sys_action)  # 186
                test_Y_orig = turn['annotation']
                data[part].append([test_X, test_Y, test_Y_orig])
                dst.state['system_action'] = sys_action

    return data


if __name__ == '__main__':
    data = build_test_state()
    sys = GDPL()
    sys.load("199")

    policy_output = {}
    seq = 0
    for test_X, _, test_Y, in data['test']:
        prd_Y = sys.predict(test_X)
        policy_output[str(seq)] = {"state": test_X,
                                   "golden_action": test_Y, "prediction": prd_Y}
        seq += 1

    print(f"测试状态数:{len(policy_output)}")
    predict_golden = []
    for _, items in policy_output.items():
        predict_golden_temp = []
        golden = []
        prediction = []
        for i in items['golden_action']:
            act = i['act_label']
            sv_list = i['slot_values']
            for j in sv_list:
                domain = j['domain'] if j['domain'] != '' else 'none'
                slot = j['slot'] if j['slot'] != '' else 'none'
                value = 'none'
                adsv = '-'.join((act, domain, slot, value))
                golden.append(adsv)
        for i in items['prediction']:
            adsv = '-'.join(i)
            prediction.append(adsv)

        predict_golden_temp.append(golden)
        predict_golden_temp.append(prediction)
        predict_golden.append(predict_golden_temp)
    precision, recall, F1 = calculateF1(predict_golden)
    print(f"precision:{precision: .6f}")
    print(f"recall:{recall: .6f}")
    print(f"F1:{F1: .6f}")

    # 生成output_file
    output_file = 'convlab2/policy/gdpl/diachat/test/policy_output.json'
    with open(output_file, 'w', encoding='UTF-8') as f:
        json.dump(policy_output, f, ensure_ascii=False, sort_keys=True, indent=4)
