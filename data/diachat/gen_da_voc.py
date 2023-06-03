import json
import zipfile
import os
from convlab2.util.crosswoz.lexicalize import delexicalize_da


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))

#act_label-domain
def gen_act_domain(data):
    act_domain = {"User":[],"Doctor":[]}
    for i, conversation in enumerate(data):
        for i, utterance in enumerate(conversation['utterances']):
            role = utterance['agentRole']

            for i, annotation in enumerate(utterance['annotation']):
                act_label = annotation['act_label']
                for i, slot_values in enumerate(annotation['slot_values']):
                    domain = slot_values['domain']
                    if domain == '':
                        domain = 'none'
                    intent_domain=act_label+'-'+domain
                    act_domain[role].append(intent_domain)
                    
    for r in act_domain.keys():                
        #去重
        act_domain[r] = list(set(act_domain[r]))
        #排序
        act_domain[r] = sorted(act_domain[r])
    return act_domain


#act_label-domain-slot
def gen_da_voc(data):
    act_domain_slot = {"User":[],"Doctor":[]}
    for i, conversation in enumerate(data):
        for i, utterance in enumerate(conversation['utterances']):
            role = utterance['agentRole']

            for i, annotation in enumerate(utterance['annotation']):
                act_label = annotation['act_label']
                for i, slot_values in enumerate(annotation['slot_values']):
                    domain = slot_values['domain']
                    slot = slot_values['slot']
                    if domain == '':
                        domain = 'none'
                    if slot =='':
                        slot = 'none'
                    intent_domain_slot=act_label+'-'+domain+'-'+slot
                    act_domain_slot[role].append(intent_domain_slot)
                    
    for r in act_domain_slot.keys():                
        #去重
        act_domain_slot[r] = list(set(act_domain_slot[r]))
        #排序
        act_domain_slot[r] = sorted(act_domain_slot[r])
    return act_domain_slot



if __name__ == '__main__':
    data = read_zipped_json('train.json.zip','train.json')
    act_domain = gen_act_domain(data)
    json.dump(act_domain['User'], open('usr_act_domain.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    json.dump(act_domain['Doctor'], open('sys_act_domain.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    
    act_domain_slot= gen_da_voc(data)
    json.dump(act_domain_slot['User'], open('usr_act_domain_slot.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    json.dump(act_domain_slot['Doctor'], open('sys_act_domain_slot.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

