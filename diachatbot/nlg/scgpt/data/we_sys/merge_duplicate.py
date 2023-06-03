# -*- coding:utf-8 -*-
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
def file_qc():
    str1 = []
    file_1 = open("sys_train2.txt","r",encoding="utf-8")
    for line in file_1.readlines():
        str1.append(line.replace("\n",""))

    str2 = []
    file_2 = open("sys_test2.txt", "r", encoding="utf-8")
    for line in file_2.readlines():
        str2.append(line.replace("\n", ""))

    str3 = []
    file_3 = open("sys_val2.txt", "r", encoding="utf-8")
    for line in file_3.readlines():
        str3.append(line.replace("\n", ""))

    str_all = set(str1 + str2 + str3)      #将两个文件放到集合里，过滤掉重复内容
    str_all=list(str_all)
    str_all = np.array(str_all)

    train, test = train_test_split(str_all, test_size=0.2, random_state=5)

    for str in train.tolist() :             #去重后的结果写入文件
            with open("final_train2.txt","a+",encoding="utf-8") as f:
                f.write(str + "\n")
    for str in test.tolist() :             #去重后的结果写入文件
            with open("final_test2.txt","a+",encoding="utf-8") as f:
                f.write(str + "\n")
if __name__=="__main__":
    file_qc()