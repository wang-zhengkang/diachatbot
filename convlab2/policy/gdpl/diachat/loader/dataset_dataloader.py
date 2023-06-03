"""
Dataloader base class. Every dataset should inherit this class and implement its own dataloader.
"""
from abc import ABC, abstractmethod
import os
from convlab2.util.file_util import read_zipped_json
from convlab2 import DATA_ROOT

from convlab2.policy.gdpl.diachat.dst import RuleDST
from convlab2.policy.gdpl.diachat.util.state_structure import *


class DatasetDataloader(ABC):
    def __init__(self):
        self.data = None

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """
        load data from file, according to what is need
        :param args:
        :param kwargs:
        :return: data
        """
        pass


class DiachatDataloader(DatasetDataloader):
    def __init__(self, zh=False):
        super(DiachatDataloader, self).__init__()
        self.zh = zh

    def load_data(self, data_dir=None, data_key="all", role="all"):
        def org(da, act_label, dsv_list: list):
            for dsv in dsv_list:
                da_temp = []
                da_temp.append(act_label)
                da_temp.append(dsv["domain"] if dsv["domain"] else "none")
                da_temp.append(dsv["slot"] if dsv["slot"] else "none")
                da_temp.append(dsv["value"] if dsv["value"] else "none")
                da.append(da_temp)

        if data_dir is None:
            data_dir = os.path.join(DATA_ROOT, "diachat")

        assert role in ["sys", "usr", "all"]
        info_list = [
            "last_sys_action",  # 上一轮的系统动作
            "sys_action",
            "usr_action",
            "belief_state",
            "cur_domain",
            "askfor_dsv",
            "askforsure_dsv",
            "terminated"
        ]

        self.data = {"train": {}, "val": {}, "test": {}}
        dst = RuleDST()
        if data_key == "all":
            data_key_list = ["train", "val", "test"]
        else:
            data_key_list = [data_key]
        for data_key in data_key_list:
            data = read_zipped_json(
                os.path.join(data_dir, "{}.json.zip".format(data_key)),
                "{}.json".format(data_key),
            )
            print("loaded {}, size {}".format(data_key, len(data)))
            for x in info_list:
                self.data[data_key][x] = []
            for session in data:
                dst.init_session()
                for i, utterance in enumerate(session["utterances"]):
                    da = []  # each turn da
                    for annotation in utterance["annotation"]:
                        act_label = annotation["act_label"]
                        dsv = annotation["slot_values"]
                        org(da, act_label, dsv)
                    if utterance["agentRole"] == "User":
                        dst.update(da)
                        self.data[data_key]["last_sys_action"].append(dst.state["sys_action"])
                        self.data[data_key]["usr_action"].append(dst.state["usr_action"])
                        self.data[data_key]["belief_state"].append(dst.state["belief_state"])
                        self.data[data_key]["cur_domain"].append(dst.state["cur_domain"])
                        self.data[data_key]["askfor_dsv"].append(dst.state["askfor_dsv"])
                        self.data[data_key]["askforsure_dsv"].append(dst.state["askforsure_dsv"])
                        if i == len(session["utterances"]) - 2:
                            self.data[data_key]["terminated"].append("Ture")
                        else:
                            self.data[data_key]["terminated"].append("False")
                    else:
                        dst.update_by_sysda(da)
                        self.data[data_key]["sys_action"].append(dst.state["sys_action"])
        return self.data
