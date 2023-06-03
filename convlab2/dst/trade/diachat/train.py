# specify cuda id
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   #获取环境变量 指定GPU

from convlab2.dst.trade.diachat.utils.config import MODE
from tqdm import tqdm
import torch.nn as nn
import shutil, zipfile
from convlab2.util.file_util import cached_path

from convlab2.dst.trade.diachat.utils.config import *
from convlab2.dst.trade.diachat.models.TRADE import *

'''
python train.py
'''
print("\n当前工作路径：{}".format(os.getcwd()))



early_stop = args['earlyStop']   #默认为 BLEU   但是下面数据集为mltiwoz时候会重置为None

if args['dataset']=='multiwoz':
    print("参数dataset为multiwoz,early_stop会设置为None")
    from convlab2.dst.trade.diachat.utils.utils_multiWOZ_DST import *
    early_stop = None
else:
    print("You need to provide the --dataset information")
    exit(1)

# specify model parameters
args['decoder'] = 'TRADE'
args['batch'] = 4
args['drop'] = 0.2
args['learn'] = 0.001
args['load_embedding'] = 1

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
# download_data()
train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq_cn(training=True, task=args['task'],
                                                                                              sequicity=False, batch_size=int(args['batch']))
                                        # SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
model = globals()[args['decoder']](        # args["decoder"]默认为TRADE
    hidden_size=int(args['hidden']), 
    lang=lang, 
    path=args['path'],
    task=args['task'], 
    lr=float(args['learn']), 
    dropout=float(args['drop']),
    slots=SLOTS_LIST,
    gating_dict=gating_dict, 
    nb_train_vocab=max_word,
    mode=MODE)

print("[Info] Slots include ", SLOTS_LIST)
print("[Info] Unpointable Slots include ", gating_dict)

for epoch in range(200):
    print("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))      # train是dataloader  <torch.utils.data.dataloader.DataLoader object at 0x000001A089AF1548>
                                                        #train的长度是10587  enumerate(train)的长度仍然是10587 但是每个元素包含batchsize=4的对话 也就是说会有重复的训练 每次运行batch随机
    for i, data in pbar:
        ## only part data to train
        # if MODE == 'cn' and i >= 1400: break
        model.train_batch(data, int(args['clip']), SLOTS_LIST[1], reset=(i==0))    #梯度剪裁的最大范数clip默认为10  # SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]   只在i==0时候初始化
        model.optimize(args['clip'])
        pbar.set_description(model.print_loss())
        # print(data)
        # exit(1)

    if((epoch+1) % int(args['evalp']) == 0):
        
        acc = model.evaluate(dev, avg_best, SLOTS_LIST[2], early_stop)
        model.scheduler.step(acc)

        if(acc >= avg_best):
            avg_best = acc
            cnt=0
            best_model = model
        else:
            cnt+=1

        if(cnt == args["patience"] or (acc==1.0 and early_stop==None)): 
            print("Ran out of patient, early stop...")  
            break 

