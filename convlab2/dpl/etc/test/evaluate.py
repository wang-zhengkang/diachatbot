import json

from convlab2.dpl.etc.util.vector_diachat import DiachatVector
from convlab2.dpl.etc.loader.build_data import build_data

from convlab2.dpl.gdpl.diachat.gdpl import GDPL
from convlab2.dpl.mle.diachat.mle import MLE


def calculateF1(predict_target_act):
    TP, FP, FN = 0, 0, 0
    for item in predict_target_act:
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
    
    precise = 1.0 * TP / (TP + FP) if (TP + FP) else 0.
    recall = 1.0 * TP / (TP + FN) if (TP + FN) else 0.
    F1 = 2.0 * precise * recall / (precise + recall) if (precise + recall) else 0.
    
    return F1, precise, recall


def test(policy_sys):
    with open('convlab2/dpl/etc/data/test.json', 'r') as f:
        source_data = json.load(f)
        test_data = build_data(source_data)
    vector = DiachatVector()
    predict_target_act = []
    for _, state_act_pair in enumerate(test_data):
        state_vec = state_act_pair[0]
        target_act_vec = state_act_pair[1]
        predict_act = policy_sys.predict(state_vec)
        target_act = vector.action_devectorize(target_act_vec)
        temp = [predict_act, target_act]
        predict_target_act.append(temp)
    F1, precise, recall = calculateF1(predict_target_act)
    return F1, precise, recall


def test(policy_sys):
    with open('convlab2/dpl/etc/data/test.json', 'r') as f:
        source_data = json.load(f)
        test_data = build_data(source_data)
    vector = DiachatVector()
    predict_target_act = []
    for _, state_act_pair in enumerate(test_data):
        state_vec = state_act_pair[0]
        target_act_vec = state_act_pair[1]
        predict_act = policy_sys.predict(state_vec)
        target_act = vector.action_devectorize(target_act_vec)
        temp = [predict_act, target_act]
        predict_target_act.append(temp)
    precise, recall, F1 = calculateF1(predict_target_act)
    return precise, recall, F1


if __name__ == '__main__':
    with open('convlab2/dpl/etc/data/test.json', 'r') as f:
        source_data = json.load(f)
        test_data = build_data(source_data)

        print(f"总共{len(test_data)}条状态进行测试")

        vector = DiachatVector()
        sys = GDPL()
        # sys = MLE(True)

        predict_target_act = []
        for i, state_act_pair in enumerate(test_data):
            state_vec = state_act_pair[0]
            target_act_vec = state_act_pair[1]
            predict_act = sys.predict(state_vec)
            target_act = vector.action_devectorize(target_act_vec)
            temp = [predict_act, target_act]
            predict_target_act.append(temp)
        F1, precise, recall = calculateF1(predict_target_act)
        print(F1, precise, recall)
