import configparser
import os
import zipfile
from copy import deepcopy
from collections import defaultdict
from pprint import pprint
import torch
import re
import sys

sys.path.append('.')
from convlab2.util.diachat.domain_act_slot import *
from convlab2.util.file_util import cached_path
from convlab2.nlg.sclstm.crosswoz.loader.dataset_woz import SimpleDatasetWoz
from convlab2.nlg.sclstm.diachat.loader.dataset_diachat import DatasetDiachat, SimpleDatasetDiachat
from convlab2.nlg.sclstm.model.lm_deep import LMDeep
from convlab2.nlg.nlg import NLG

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "nlg_sclstm_crosswoz.zip")


# act_labels={
#     "noargs_act":['Accept','AskWhy','Assure','Deny','GeneralAdvice','Chitchat','GeneralExplanation'],
#     "doctor_act":['Advice','AdviceNot','Explanation','AskForSure','AskFor','AskHow','Assure','Deny','GeneralAdvice','Chitchat','GeneralExplanation'],
#     "user_act":['Inform','Accept','AskForSure','AskWhy','AskFor','AskHow','Assure','Deny','Uncertain','Chitchat'],
#     "all_act":['Inform','Advice','AdviceNot','Accept','Explanation','AskForSure','AskWhy','AskFor','AskHow','Assure','Deny','Uncertain','GeneralAdvice','Chitchat','GeneralExplanation']
#     }

def act_no_args(act):
    return act in act_labels['noargs_act']

def parse(is_user):
    if is_user:
        args = {
            'model_path': CURRENT_DIR + 'sclstm_usr0.pt',
            'n_layer': 2,
            'beam_size': 10
        }
    else:
        args = {
            'model_path': CURRENT_DIR + 'sclstm-0509.pt',
            'n_layer': 4,
            'beam_size': 10
        }

    config = configparser.ConfigParser()
    if is_user:
        config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/config_usr.cfg'))
    else:
        config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/config.cfg'))
    config.set('DATA', 'dir', os.path.dirname(os.path.abspath(__file__)))

    return args, config

'''
add by yangjinfeng
Tsung-Hsien Wen, Milica Gašić, Nikola Mrkšić, Pei-Hao Su, David Vandyke, Steve Young. 2015.
Semantically conditioned LSTM-based natural language generation for spoken dialogue systems.
In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing,
pages 1711–1721, Lisbon, Portugal. Association for Computational Linguistics.
https://aclanthology.org/D15-1199.pdf
'''
class SCLSTM(NLG):
    def __init__(self,
                 use_cuda=True,
                 is_user=False):

#         print(archive_file)
#         if not os.path.isfile(archive_file):
#             if not model_file:
#                 raise Exception("No model for SC-LSTM is specified!")
#             #缓存到这里了  C:\Users\admin\.convlab2\cache
#             archive_file = cached_path(model_file)
#         model_dir = os.path.dirname(os.path.abspath(__file__))
#         if not os.path.exists(os.path.join(model_dir, 'resource')):
#             archive = zipfile.ZipFile(archive_file, 'r')
#             archive.extractall(model_dir)

        self.USE_CUDA = use_cuda
        self.args, self.config = parse(is_user)
        self.dataset = SimpleDatasetWoz(self.config)

        # get model hyper-parameters
        hidden_size = self.config.getint('MODEL', 'hidden_size')

        # get feat size
        d_size = self.dataset.do_size + self.dataset.da_size + self.dataset.sv_size  # len of 1-hot feat
        vocab_size = len(self.dataset.word2index)

        self.model = LMDeep('sclstm', vocab_size, vocab_size, hidden_size, d_size, n_layer=self.args['n_layer'],
                            use_cuda=use_cuda)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.args['model_path'])
        # print(model_path)
        assert os.path.isfile(model_path)
#         self.model.load_state_dict(torch.load(model_path))
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
#         self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cuda:0')))
        self.model.eval()
#         for name, param in self.model.named_parameters():
#             print(name, param.shape, param.device, param.requires_grad)
        if use_cuda:
            self.model.cuda()

    def generate_delex(self, meta):
        """
        meta = [
        {"act_label":"Inform",
         "slot_values":[{"domain":"饮食","slot":"饮食名","value":"莲藕排骨"},{"domain":"饮食","slot":"饮食名","value":"汤"}]
         },
        {"act_label":"Inform",
         "slot_values":[{"domain":"问题","slot":"血糖值","value":"升高"}]
         },
        {"act_label":"AskForSure",
         "slot_values":[{"domain":"行为","slot":"行为名","value":"喝"}]
         }
        ]
        """
        intent_list = []
        intent_frequency = defaultdict(int)
        feat_dict = dict()
        for each_act in meta:
            act = deepcopy(each_act)

            for ele in act['slot_values']:#slot_values对应的列表中可能有多个字典
                intent = '+'.join([act['act_label'],ele['domain'],ele['slot']])
                intent_list.append(intent)
                intent_frequency[intent] += 1
            # content replacement
                value = 'none'
                freq = 'none'
                if not act_no_args(act['act_label']):
                    if act['act_label'] == 'AskFor':
                        freq = '?'
                        value = '?'
                    else:    
                        value = ele['value']
                        freq = str(intent_frequency[intent])

                # generate the formation in feat.json
                new_act = intent.split('+')
                feat_value = []
                if act_no_args(new_act[0]):#如果act 没有arguments
                    if new_act[0] in ['GeneralAdvice','GeneralExplanation']:
                        feat_key = new_act[1] + '-' + new_act[0]
                        feat_value = ['none','none','none']
                    else:                                
                        feat_key = 'General-'+new_act[0]
                        feat_value = ['none','none','none']
                else:
                    feat_key = new_act[1] + '-' + new_act[0]
                    feat_value = [new_act[2], freq, value]

                feat_dict[feat_key] = feat_dict.get(feat_key, [])
                feat_dict[feat_key].append(feat_value)

        meta = deepcopy(feat_dict)

        # remove invalid dialog act
        meta_ = deepcopy(meta)
        for k, v in meta.items():
            for triple in v:
                voc = 'd-a-s-v:' + k + '-' + triple[0] + '-' + triple[1]
                if voc not in self.dataset.cardinality:
                    meta_[k].remove(triple)
            if not meta_[k]:
                del (meta_[k])
        meta = meta_

        # mapping the inputs
        do_idx, da_idx, sv_idx, featStr = self.dataset.getFeatIdx(meta)
        do_cond = [1 if i in do_idx else 0 for i in range(self.dataset.do_size)]  # domain condition
        da_cond = [1 if i in da_idx else 0 for i in range(self.dataset.da_size)]  # dial act condition
        sv_cond = [1 if i in sv_idx else 0 for i in range(self.dataset.sv_size)]  # slot/value condition
        feats = [do_cond + da_cond + sv_cond]

        feats_var = torch.FloatTensor(feats)
        if self.USE_CUDA:
            feats_var = feats_var.cuda()

        decoded_words = self.model.generate(self.dataset, feats_var, self.args['beam_size'])
        delex = decoded_words[0]  # (beam_size)

        return delex

    def generate_slots(self, meta):
        meta = deepcopy(meta)

        delex = self.generate_delex(meta)
        # get all informable or requestable slots
        slots = []
        for sen in delex:
            slot = []
            counter = {}
            words = sen.split()
            for word in words:
                if word.startswith('slot-'):
                    placeholder = word[5:]
                    if placeholder not in counter:
                        counter[placeholder] = 1
                    else:
                        counter[placeholder] += 1
                    slot.append(placeholder + '-' + str(counter[placeholder]))
            slots.append(slot)

        # for i in range(self.args.beam_size):
        #     print(i, slots[i])

        return slots[0]

    def _value_replace(self, sentences, dialog_act):
        ori_sen = deepcopy(sentences)
        dialog_act = deepcopy(dialog_act)
        intent_frequency = defaultdict(int)
        for each_act in dialog_act:
            act = deepcopy(each_act)
            for ele in act['slot_values']:#slot_values对应的列表中可能有多个字典

                intent = '+'.join([act['act_label'],ele['domain'],ele['slot']])
                intent_frequency[intent] += 1
                if intent_frequency[intent] > 1:  # if multiple same intents...
                    intent += str(intent_frequency[intent])

                value = ele['value']
                sentences = sentences.replace('[' + intent + ']', value)
                sentences = sentences.replace('[' + intent + '1]', value)  # if multiple same intents and this is 1st

        return sentences

    def _prepare_intent_string(self, cur_act):
        """
        Generate the intent form **to be used in selecting templates** (rather than value replacement)
        :param cur_act: one act list
        :return: one intent string
        """

        cur_act = deepcopy(cur_act)
        if cur_act[0] == 'Inform' and '酒店设施' in cur_act[2]:
            cur_act[2] = cur_act[2].split('-')[0] + '+' + cur_act[3]
        elif cur_act[0] == 'Request' and '酒店设施' in cur_act[2]:
            cur_act[2] = cur_act[2].split('-')[0]
        if cur_act[0] == 'Select':
            cur_act[2] = '源领域+' + cur_act[3]
        try:
            if '+'.join(cur_act) == 'Inform+景点+门票+免费':
                intent = '+'.join(cur_act)
            # "Inform+景点+周边酒店+无"
            elif cur_act[3] == '无':
                intent = '+'.join(cur_act)
            else:
                intent = '+'.join(cur_act[:-1])
        except Exception as e:
            print('Act causing error:')
            pprint(cur_act)
            raise e
        return intent

    def generate(self, meta):
#         meta = [[str(x[0]), str(x[1]), str(x[2]), str(x[3]).lower()] for x in meta]
        meta = deepcopy(meta)

        delex = self.generate_delex(meta)
#         print(delex)

        for dx in delex:
            gen = self._value_replace(dx.replace('UNK_token', '').replace(' ', ''), meta)
#             print(gen)
        return self._value_replace(delex[0].replace('UNK_token', '').replace(' ', ''), meta)


if __name__ == '__main__':
    model_sys = SCLSTM(is_user=False, use_cuda=True)
#     model_sys = SCLSTM(is_user=True, use_cuda=False)
    inputaction=[
#         {"act_label":"Inform",
#          "slot_values":[{"domain":"饮食","slot":"饮食名","value":"莲藕排骨"},{"domain":"饮食","slot":"饮食名","value":"汤"}]
#          },
#         {"act_label":"Inform",
#          "slot_values":[{"domain":"问题","slot":"血糖值","value":"升高"}]
#          },
#         {"act_label":"AskForSure",
#          "slot_values":[{"domain":"行为","slot":"行为名","value":"喝"}]
#          }
            {
            "act_label": "AskFor",
            "slot_values": [{"domain": "问题","slot": "血糖值","value":"?"}]
            }
        ]
#     print(model_sys.generate([['Inform', '餐馆', '人均消费', '100-150元'], ['Request', '餐馆', '电话', '']]))
    print(model_sys.generate(inputaction))
