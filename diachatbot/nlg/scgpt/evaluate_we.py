from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import json
import jieba
smooth = SmoothingFunction()  # 定义平滑函数对象
# labels = [['我', '是', '狗']]
# predicts = [['我', '是', '猫','狗']]
# corpus_score_2 = corpus_bleu(labels, predicts, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
# corpus_score_3 = corpus_bleu(labels, predicts, weights=(0, 1, 0, 0), smoothing_function=smooth.method1)
# corpus_score_4 = corpus_bleu(labels, predicts, weights=(0, 0, 1, 0), smoothing_function=smooth.method1)
# print(corpus_score_2)
# print(corpus_score_3)
# print(corpus_score_4)


with open('final_result2.json','r', encoding='utf8') as f_load:
    final_result_we1 = json.load(f_load)
scores_1=[]
scores_2=[]
scores_3=[]
scores_4=[]
scores_5=[]
for i in range(0,len(final_result_we1)):
    #读取
    target = final_result_we1[i][1].replace(' ','')
    predict =final_result_we1[i][2].replace(' ','')
    #转为列表
    target=[list(target)]
    predict=[list(predict)]
    print(target)
    corpus_score_1 = corpus_bleu(target, predict, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
    corpus_score_2 = corpus_bleu(target, predict, weights=(0, 1, 0, 0), smoothing_function=smooth.method1)
    corpus_score_3 = corpus_bleu(target, predict, weights=(0, 0, 1, 0), smoothing_function=smooth.method1)
    corpus_score_4 = corpus_bleu(target, predict, weights=(0, 0, 0, 1), smoothing_function=smooth.method1)
    corpus_score_5 = corpus_bleu(target, predict, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    scores_1.append(corpus_score_1)
    scores_2.append(corpus_score_2)
    scores_3.append(corpus_score_3)
    scores_4.append(corpus_score_4)
    scores_5.append(corpus_score_5)
print(scores_1)
bleu_1=0
bleu_2=0
bleu_3=0
bleu_4=0
bleu_5=0
for i in range(0,len(scores_1)):
    bleu_1 = bleu_1 + scores_1[i]
    bleu_2 = bleu_2 + scores_2[i]
    bleu_3 = bleu_3 + scores_3[i]
    bleu_4 = bleu_4 + scores_4[i]
    bleu_5 = bleu_5 + scores_5[i]
print('BLEU1值',bleu_1/len(scores_1))
print('BLEU2值',bleu_2/len(scores_2))
print('BLEU3值',bleu_3/len(scores_3))
print('BLEU4值',bleu_4/len(scores_4))


