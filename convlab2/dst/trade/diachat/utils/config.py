import os
import logging 
import argparse
from tqdm import tqdm
import torch

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

MODE = 'cn'
data_version = 'init'  # processed

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

MAX_LENGTH = 10
'''
ArgumentParser 对象:
    prog - 程序的名称（默认：sys.argv[0]）
    usage - 描述程序用途的字符串（默认值：从添加到解析器的参数生成）
    description - 在参数帮助文档之前显示的文本（默认值：无）
    epilog - 在参数帮助文档之后显示的文本（默认值：无）
    parents - 一个 ArgumentParser 对象的列表，它们的参数也应包含在内
    formatter_class - 用于自定义帮助文档输出格式的类
    prefix_chars - 可选参数的前缀字符集合（默认值：’-’）
    fromfile_prefix_chars - 当需要从文件中读取其他参数时，用于标识文件名的前缀字符集合（默认值：None）
    argument_default - 参数的全局默认值（默认值： None）
    conflict_handler - 解决冲突选项的策略（通常是不必要的）
    add_help - 为解析器添加一个 -h/–help 选项（默认值： True）
    allow_abbrev - 如果缩写是无歧义的，则允许缩写长选项 （默认值：True）
add_argument() 方法：
    name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
    action - 当参数在命令行中出现时使用的动作基本类型。
    nargs - 命令行参数应当消耗的数目。
    const - 被一些 action 和 nargs 选择所需求的常数。
    default - 当参数未在命令行中出现时使用的值。
    type - 命令行参数应当被转换成的类型。
    choices - 可用的参数的容器。
    required - 此命令行选项是否可省略 （仅选项可用）。
    help - 一个此选项作用的简单描述。
    metavar - 在使用方法消息中使用的参数值示例。
    dest - 被添加到 parse_args() 所返回对象上的属性名。
————————————————

原文链接：https://blog.csdn.net/lizhiyuanbest/article/details/104975848
'''
parser = argparse.ArgumentParser(description='TRADE Multi-Domain DST')

# Training Setting
parser.add_argument('-ds','--dataset', help='dataset', required=False, default="multiwoz")  # dataset 默认 multiwoz
parser.add_argument('-t','--task', help='Task Number', required=False, default="dst")       # task默认 dst
parser.add_argument('-path','--path', help='path of the file to load', required=False)
parser.add_argument('-sample','--sample', help='Number of Samples', required=False,default=None)
parser.add_argument('-patience','--patience', help='', required=False, default=60, type=int)    # earlystop的参数 patience默认 6  能够容忍多少个epoch内都没有improvement。
parser.add_argument('-es','--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')    # earlyStop 评价标准默认 BLEU  可以达到当训练集上的loss不在减小（即减小的程度小于某个阈值）的时候停止继续训练。
parser.add_argument('-all_vocab','--all_vocab', help='', required=False, default=1, type=int)     # all_vocab 默认 1
parser.add_argument('-imbsamp','--imbalance_sampler', help='', required=False, default=0, type=int)  # imbalance_sampler 不平衡数据采样 默认0
parser.add_argument('-data_ratio','--data_ratio', help='', required=False, default=100, type=int)    # data_ratio 数据比率  默认100
parser.add_argument('-um','--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)   # unk_mask
parser.add_argument('-bsz','--batch', help='Batch_size', required=False, type=int)

# Testing Setting
parser.add_argument('-rundev','--run_dev_testing', help='', required=False, default=0, type=int)
parser.add_argument('-viz','--vizualization', help='vizualization', type=int, required=False, default=0)
## model predictions
parser.add_argument('-gs','--genSample', help='Generate Sample', type=int, required=False, default=1)  #### change this when testing
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-an','--addName', help='An add name for the model folder', required=False, default='')
parser.add_argument('-eb','--eval_batch', help='Evaluation Batch_size', required=False, type=int, default=0)

# Model architecture
parser.add_argument('-gate','--use_gate', help='', required=False, default=1, type=int)
parser.add_argument('-le','--load_embedding', help='', required=False, default=0, type=int)
parser.add_argument('-femb','--fix_embedding', help='', required=False, default=0, type=int)
parser.add_argument('-paral','--parallel_decode', help='', required=False, default=0, type=int)

# Model Hyper-Parameters
parser.add_argument('-dec','--decoder', help='decoder model', required=False)
parser.add_argument('-hdd','--hidden', help='Hidden size', required=False, type=int, default=100)
parser.add_argument('-lr','--learn', help='Learning Rate', required=False, type=float)
parser.add_argument('-dr','--drop', help='Drop Out', required=False, type=float)
parser.add_argument('-lm','--limit', help='Word Limit', required=False,default=-10000)
parser.add_argument('-clip','--clip', help='gradient clipping', required=False, default=10, type=int) 
parser.add_argument('-tfr','--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False, default=0.5)
# parser.add_argument('-l','--layer', help='Layer Number', required=False)

# Unseen Domain Setting
parser.add_argument('-l_ewc','--lambda_ewc', help='regularization term for EWC loss', type=float, required=False, default=0.01)
parser.add_argument('-fisher_sample','--fisher_sample', help='number of sample used to approximate fisher mat', type=int, required=False, default=0)
parser.add_argument("--all_model", action="store_true")
parser.add_argument("--domain_as_task", action="store_true")
parser.add_argument('--run_except_4d', help='', required=False, default=1, type=int)
parser.add_argument("--strict_domain", action="store_true")
parser.add_argument('-exceptd','--except_domain', help='', required=False, default="", type=str)
parser.add_argument('-onlyd','--only_domain', help='', required=False, default="", type=str)

print(parser.parse_known_args())
# vars()返回对象object的属性和属性值的字典对象
args = vars(parser.parse_known_args(args=[])[0])  # 默认args=none  所以可以省略args=[]
if args["load_embedding"]:
    args["hidden"] = 100
if args["fix_embedding"]:
    args["addName"] += "FixEmb"
if args["except_domain"] != "":
    args["addName"] += "Except"+args["except_domain"]
if args["only_domain"] != "":
    args["addName"] += "Only"+args["only_domain"]

