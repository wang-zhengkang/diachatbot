# 将prediction和golden的act domain slot进行对比，不考虑value
import json
import pprint


def returnSum(myDict):
    sum = 0
    for i in myDict:
        sum = sum + myDict[i]

    return sum


if __name__ == '__main__':
    with open('./policy_output.json', 'r', encoding='utf-8') as fp:
        json_data = json.load(fp)
        print("测试状态数：", len(json_data))
        total = 0
        correct = 0
        for _, items in json_data.items():
            golden_dic = {}
            golden_action = items['golden_action']
            prediction = items['prediction']
            for g_a in golden_action:
                act = g_a['act_label']
                for g_sv in g_a['slot_values']:
                    domain = g_sv['domain']
                    slot = g_sv['slot']
                    ads = act + domain + slot
                    if ads not in golden_dic:
                        golden_dic[ads] = 1
                    else:
                        golden_dic[ads] += 1
            total += returnSum(golden_dic)
            for p_a in prediction:
                act = p_a['act_label']
                for p_sv in p_a['slot_values']:
                    domain = p_sv['domain']
                    slot = p_sv['slot']
                    ads = act + domain + slot
                    if ads in golden_dic:
                        if golden_dic[ads] == 0:
                            break
                        correct += 1
                        golden_dic[ads] -= 1

        print(f'模型预测act_domain_slot匹配数量：{correct}, 测试act_domain_slot总数：{total}')
        print(f'正确率：{correct / total}')