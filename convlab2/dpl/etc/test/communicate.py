from convlab2.dpl.gdpl.diachat.gdpl import GDPL
from convlab2.dpl.mle.diachat.mle import MLE
from convlab2.dpl.etc.util.dst import RuleDST
from convlab2.dpl.vhus.diachat_DG_BiGRU.vhus_diachat import UserPolicyVHUS
from pprint import pprint


if __name__ == '__main__':
    # 加载dst sys usr
    dst = RuleDST()
    dst.init_session()
    # sys = GDPL()
    sys = MLE(True)
    usr = UserPolicyVHUS(is_load_model=True)
    usr.init_session()

    print("*"*30 + "Goal" + "*"*30)
    pprint(usr.goal)
    start_tag = True

    while True:
        # usr发起对话并更新状态
        usr_action = usr.predict() if start_tag else usr.predict(sys_action)
        start_tag = False
        print("*"*30 + "usr_action" + "*"*30)
        pprint(usr_action)
        dst.update(usr_action)
        

        # sys根据状态做出回应并更新对话
        sys_action = sys.predict(dst.state)
        print("*"*30 + "sys_action" + "*"*30)
        pprint(sys_action)
        dst.update_by_sysda(sys_action)
        
        # 判断对话是否结束
        if usr.terminated:
            break
    