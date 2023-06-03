# -*- coding: utf-8 -*-
import os
import json
from convlab2.nlg.scgpt.utils import dict2dict, dict2seq

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "val/")

with open(os.path.join(data_dir, 'val.json'),"r", encoding='utf8') as f:
    data = json.load(f)

sys_results = {}
user_results = {}
i=0

for sess in data:
    i = i+1

    utterances = sess['utterances']
    print(utterances)
    print("---------------------")
    for diags in utterances:
        i = i+1
        user_turns = []
        sys_turns= []
        turn = {'text': ''}
        if diags['annotation'] != '':
            if diags['agentRole'] == 'User':
                turn['text'] = diags['utterance']
                turn['da'] = ''
                turn['slots'] = ''
                for diag in diags['annotation']:
                    i = i + 1
                    if turn['da'] == '':
                        turn['da'] = diag['act_label']
                    else:
                        turn['da'] = turn['da'] + ' @ ' + diag['act_label']
                    for slot_values in diag['slot_values']:
                        i = i + 1
                        slot_value = slot_values['domain'] + '.' + slot_values['slot'] + ' = ' + slot_values['value'] + ' ; '
                        turn['slots'] += slot_value
                user_turns.append(turn)
                user_results[i] = user_turns
            else:
                turn['text'] = diags['utterance']
                turn['da'] = ''
                turn['slots'] = ''
                for diag in diags['annotation']:
                    i = i + 1
                    if turn['da'] == '':
                        turn['da'] = diag['act_label']
                    else:
                        turn['da'] = turn['da'] + ' @ ' + diag['act_label']
                    for slot_values in diag['slot_values']:
                        i = i + 1
                        slot_value = slot_values['domain'] + '.' + slot_values['slot'] + ' = ' + slot_values['value'] + ' ; '
                        turn['slots'] += slot_value
                sys_turns.append(turn)
                sys_results[i] = sys_turns


def write_file(name, data):
    with open(f'{name}.txt', 'w', encoding='utf-8') as f:
        for ID in data:
            sess = data[ID]
            for turn in sess:
                # if not turn['usr_da']:
                #     continue
                # turn['usr_da'] = eval(str(turn['usr_da']).replace('Bus','Train'))
                # da_seq = dict2seq(dict2dict(turn['usr_da'])).replace('&', 'and')
                # domains = set([key.split('-')[0] for key in turn['usr_da'].keys()])
                # for domain in domains:
                #     if domain not in ['general', 'Booking'] and not sess_domains[domain]:
                #         da_seq = da_seq.replace(domain.lower(), domain.lower()+' *', 1)
                #         sess_domains[domain] = True
                slots = turn['slots']
                slots_list = list(slots)
                slots_list.pop()
                slots_list.pop()
                slots = ''.join(slots_list)
                da_seq = ''
                da_seq += (turn['da'] + ' ( ' + slots + ')')
                da_uttr = turn['text']
                f.write(f'{da_seq} & {da_uttr}\n')

if not os.path.exists(os.path.join(cur_dir,'data')):
    os.makedirs(os.path.join(cur_dir, 'data'))
write_file(os.path.join(cur_dir, 'data/sys_val2'), sys_results)
write_file(os.path.join(cur_dir, 'data/user_val2'), user_results)
