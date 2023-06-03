'''
Created on 2022年3月17日

@author: yangjinfeng
'''
from transformers import BertTokenizer

BubianQuanjiao = '，？；：！（）'


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        if uchar in BubianQuanjiao:
            rstring += uchar
            continue
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

def tk(ut):
    utterance=strQ2B(ut)
    print(utterance)
    tokens = tokenizer.tokenize(utterance)
    print(tokens)

tk('１６２厘米，120斤。!！')
tk('空腹小于7，餐后2小时小于10，就算达标')
tk('可以，电话是80495616。')
tk("空腹５．５左右")
tk("没吃药，就是觉得瘦得太快了，三个半月，瘦了２６斤了")
tk("品牌是nvidia")
tk("用量是120mg每天")
tk("用量是120mg每天")
tk('AskForSure')

    
