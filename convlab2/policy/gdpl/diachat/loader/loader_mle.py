import os
import pickle
import torch
import torch.utils.data as data
from convlab2.policy.vector.dataset import ActDataset

from convlab2.policy.gdpl.diachat.util.state_structure import default_state
from convlab2.policy.gdpl.diachat.loader.dataset_dataloader import DiachatDataloader
from convlab2.policy.gdpl.diachat.loader.module_dataloader import ActPolicyDataloader


class ActMLEPolicyDataLoader:
    def __init__(self):
        self.vector = None

    def _build_data(self, processed_dir):
        self.data = {}
        data_loader = ActPolicyDataloader(dataset_dataloader=DiachatDataloader())
        for part in ["train", "val", "test"]:
            self.data[part] = []
            raw_data = data_loader.load_data(data_key=part, role="sys")[part]
            for (
                last_sys_action,
                sys_action,
                usr_action,
                belief_state,
                cur_domain,
                askfor_dsv,
                askforsure_dsv,
                terminated,
            ) in zip(
                raw_data["last_sys_action"],
                raw_data["sys_action"],
                raw_data["usr_action"],
                raw_data["belief_state"],
                raw_data["cur_domain"],
                raw_data["askfor_dsv"],
                raw_data["askforsure_dsv"],
                raw_data["terminated"],
            ):
                state = default_state()
                state['sys_action'] = last_sys_action
                state['usr_action'] = usr_action
                state['belief_state'] = belief_state
                state['cur_domain'] = cur_domain
                state['askfor_dsv'] = askfor_dsv
                state['askforsure_dsv'] = askforsure_dsv
                state['terminated'] = terminated
                action = sys_action
                self.data[part].append([self.vector.state_vectorize(state),
                         self.vector.action_vectorize(action)])
        for part in ["train", "val", "test"]:
            with open(os.path.join(processed_dir, "{}.pkl".format(part)), "wb") as f:
                pickle.dump(self.data[part], f)

    def _load_data(self, processed_dir):
        self.data = {}
        for part in ["train", "val", "test"]:
            with open(os.path.join(processed_dir, "{}.pkl".format(part)), "rb") as f:
                self.data[part] = pickle.load(f)

    def create_dataset(self, part, batchsz):
        print("Start creating {} dataset".format(part))
        s = []
        a = []
        for item in self.data[part]:
            s.append(torch.Tensor(item[0]))
            a.append(torch.Tensor(item[1]))
        s = torch.stack(s)
        a = torch.stack(a)
        dataset = ActDataset(s, a)
        dataloader = data.DataLoader(dataset, batchsz, True)
        print("Finish creating {} dataset".format(part))
        return dataloader
