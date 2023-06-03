from convlab2.dpl.vhus.diachat_DG_BiGRU.vhus_diachat import UserPolicyVHUS
from pprint import pprint


def str2list(action: list):
    action = list(action)
    action_list = []
    action_tmp = []
    letter_list = []
    flag = False
    for _, letter in enumerate(action):
        if letter == '\'':
            flag = not flag
            if flag:
                continue
            if not flag:
                word = ''.join(letter_list)
                action_tmp.append(word)
                letter_list = []
                if len(action_tmp) % 4 == 0:
                    action_list.append(action_tmp)
                    action_tmp = []
        if not flag:
            continue
        else:
            letter_list.append(letter)
    return action_list


if __name__ == '__main__':
    usr = UserPolicyVHUS(True)
    usr.init_session()
    print('-'*30 + 'usr initial goal' + '-'*30)
    pprint(usr.goal)
    while True:
        print('-'*30 + 'usr action' + '-'*30)
        pprint(usr.predict())
        if usr.terminated:
            print("usr端终止对话！")
            break
        # print('-'*30 + 'usr goal' + '-'*30)
        # pprint(usr.goal)

        print('-'*30 + 'Input sys action' + '-'*30)
        sys_action = str
        exec("sys_action = input()")
        if sys_action == 'stop':
            print("sys端提前终止对话！")
            break
        sys_action = str2list(sys_action)

[['AskFor', '问题', '血糖值', '?']]
[['AskFor', '治疗', '治疗项', '?']]
[['Chitchat', 'none', 'none', 'none']]
[["Advice", "行为", "行为名", "泡脚"], ["Advice", "行为", "行为名", "抹点维生素E乳"]]
[['AskHow', '行为', '行为名', '血糖控制']]
[['Explanation', '基本信息', '体重', '瘦']]
