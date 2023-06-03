"""
Evaluate NLU models on specified dataset
Usage: python evaluate.py [MultiWOZ|CrossWOZ] [TRADE|mdbt|sumbt|rule]
"""
import random
import numpy
import torch
from convlab2.dst.trade.crosswoz.trade import CrossWOZTRADE


def format_history(context):
    history = []
    for i in range(len(context)):
        history.append(['system' if i%2==1 else 'user', context[i]])
    return history

#{'Joint Acc': 0.24, 'Turn Acc': 0.9777500000000026, 'Joint F1': 0.7946916816516827}
if __name__ == '__main__':
    seed = 2020
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    model = CrossWOZTRADE('model/TRADE-multiwozdst/HDD100BSZ4DR0.2ACC-0.3228')
    model.evaluate()