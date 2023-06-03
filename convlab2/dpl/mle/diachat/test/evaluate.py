import os
import json
import zipfile
from convlab2.dpl.mle.diachat.mle import MLE
from convlab2.policy.gdpl.diachat.gdpl import GDPL
from convlab2.dpl.etc.util.state_structure import *
from convlab2.dpl.etc.util.dst import RuleDST

from convlab2.dpl.etc.util.vector_diachat import DiachatVector


def org(da, act_label, dsv_list: list):
    for dsv in dsv_list:
        da_temp = []
        da_temp.append(act_label)
        da_temp.append(dsv["domain"] if dsv["domain"] else "none")
        da_temp.append(dsv["slot"] if dsv["slot"] else "none")
        da_temp.append(dsv["value"] if dsv["value"] else "none")
        da.append(da_temp)


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


def build_test_state():
    vector = DiachatVector()
    part = 'test'
    raw_data = {}
    data = {}
    data[part] = []
    archive = zipfile.ZipFile('convlab2/dpl/etc/data/test.json.zip', 'r')
    with archive.open('{}.json'.format(part), 'r') as f:
        raw_data[part] = json.load(f)
    dst = RuleDST()
    for conversation in raw_data[part]:
        dst.init_session()
        for i, utterance in enumerate(conversation["utterances"]):
            da = []  # each turn da
            for annotation in utterance["annotation"]:
                act_label = annotation["act_label"]
                dsv = annotation["slot_values"]
                org(da, act_label, dsv)
            if utterance["agentRole"] == "User":
                dst.update(da)
                last_sys_da = dst.state["sys_da"]
                usr_da = dst.state["usr_da"]
                cur_domain = dst.state["cur_domain"]
                askfor_ds = dst.state["askfor_ds"]
                askforsure_ds = dst.state["askforsure_ds"]
                belief_state = dst.state["belief_state"]
                if i == len(conversation["utterances"]) - 2:
                    terminate = True
                else:
                    terminate = False
            else:
                dst.update_by_sysda(da)
                sys_da = dst.state["sys_da"]

                state = default_state()

                state['sys_da'] = last_sys_da
                state['usr_da'] = usr_da
                state['cur_domain'] = cur_domain
                state['askfor_ds'] = askfor_ds
                state['askforsure_ds'] = askforsure_ds
                state['belief_state'] = belief_state
                state['terminate'] = terminate
                action = sys_da
                data[part].append([vector.state_vectorize(state),
                                   vector.action_vectorize(action)])
    return data


if __name__ == '__main__':
    data = build_test_state()
    vector = DiachatVector()
    sys = MLE()

    policy_output = {}
    seq = 0
    for (test_X, test_Y) in data['test']:
        prd_Y = sys.predict(test_X)
        policy_output[str(seq)] = {"golden_action": vector.action_devectorize(test_Y),
                                   "prediction": prd_Y}
        seq += 1

    # print(f"测试状态数:{len(policy_output)}")
    # predict_golden = []
    # for _, items in policy_output.items():
    #     predict_golden_temp = []
    #     golden = []
    #     prediction = []
    #     for i in items['golden_action']:
    #         act = i['act_label']
    #         sv_list = i['slot_values']
    #         for j in sv_list:
    #             domain = j['domain'] if j['domain'] != '' else 'none'
    #             slot = j['slot'] if j['slot'] != '' else 'none'
    #             value = 'none'
    #             adsv = '-'.join((act, domain, slot, value))
    #             golden.append(adsv)
    #     for i in items['prediction']:
    #         adsv = '-'.join(i)
    #         prediction.append(adsv)

    #     predict_golden_temp.append(golden)
    #     predict_golden_temp.append(prediction)
    #     predict_golden.append(predict_golden_temp)
    # precision, recall, F1 = calculateF1(predict_golden)
    # print(f"precision:{precision: .6f}")
    # print(f"recall:{recall: .6f}")
    # print(f"F1:{F1: .6f}")

    # 生成output_file
    output_file = 'convlab2/dpl/mle/diachat/test/policy_output.json'
    with open(output_file, 'w', encoding='UTF-8') as f:
        json.dump(policy_output, f, ensure_ascii=False,
                  sort_keys=True, indent=4)
