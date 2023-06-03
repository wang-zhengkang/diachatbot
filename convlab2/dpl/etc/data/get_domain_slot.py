import json
from convlab2.dpl.etc.util.domain_act_slot import domains
from pprint import pprint

with open('convlab2/dpl/etc/data/source_data.json', 'r', encoding='UTF-8') as fp:
    full_data = json.load(fp)
    print(domains)
    domain_slot = {i: [] for i in domains}
    for session in full_data:
        for utterance in session["utterances"]:
            for act_domain_slot_dic in utterance["annotation"]:
                for sv in act_domain_slot_dic["slot_values"]:
                    domain = sv["domain"]
                    slot = sv["slot"]
                    try:
                        if slot not in domain_slot[domain]:
                            domain_slot[domain].append(slot)
                    except:
                        pass
    
    pprint(domain_slot)