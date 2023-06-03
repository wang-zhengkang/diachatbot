from convlab2.dpl.etc.util.dst import RuleDST
from convlab2.dpl.etc.util.state_structure import *
from convlab2.dpl.etc.util.vector_diachat import DiachatVector
from copy import deepcopy


def build_data(partdata):
    def org(act_label, dsv_list: list):
        da = []
        for dsv in dsv_list:
            da_temp = []
            da_temp.append(act_label)
            da_temp.append(dsv["domain"] if dsv["domain"] else "none")
            da_temp.append(dsv["slot"] if dsv["slot"] else "none")
            da_temp.append(dsv["value"] if dsv["value"] else "none")
            da.append(da_temp)
        return da
    targetdata = []
    vector = DiachatVector()
    dst = RuleDST()
    for session in partdata:
        dst.init_session()
        for i, utterance in enumerate(session["utterances"]):
            da = []
            for annotation in utterance["annotation"]:
                act_label = annotation["act_label"]
                dsv = annotation["slot_values"]
                da_tmep = org(act_label, dsv)
                for temp in da_tmep:
                    da.append(temp)
            if utterance["agentRole"] == "User":
                dst.update(da)
                if i == len(session["utterances"]) - 2:
                    dst.state['terminate'] = True
            else:
                # state不泄露
                # state_vec = vector.state_vectorize(dst.state)
                # dst.update_by_sysda(da)
                # action = dst.state["sys_da"]
                # targetdata.append([state_vec, vector.action_vectorize(action)])

                # state泄露
                state = default_state()
                state['sys_da'] = dst.state["sys_da"]
                state['usr_da'] = dst.state["usr_da"]
                state['cur_domain'] = dst.state["cur_domain"]
                state['inform_ds'] = dst.state["inform_ds"]
                state['askhow_ds'] = dst.state["askhow_ds"]
                state['askwhy_ds'] = dst.state["askwhy_ds"]
                state['askfor_ds'] = dst.state["askfor_ds"]
                state['askforsure_ds'] = dst.state["askforsure_ds"]
                state['belief_state'] = dst.state["belief_state"]
                state['terminate'] = dst.state["terminate"]
                dst.update_by_sysda(da)
                action = dst.state["sys_da"]
                targetdata.append([vector.state_vectorize(state),
                                   vector.action_vectorize(action)])
    return targetdata
