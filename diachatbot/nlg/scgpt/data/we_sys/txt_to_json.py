import json

f=open('final_test2.txt', 'r', encoding='utf-8')
files = f.readlines()
output_tests = []
i = 0
for file in files:
    i=i+1
    examples= []
    act_yuyan = file.split(' & ',1)
    act=act_yuyan[0]+','
    yuyan=act_yuyan[1].strip()+','
    examples.append(act)
    examples.append(yuyan)
    examples.append(act_yuyan[1].strip())
    output_tests.append(examples)
    print(i)
with open('final_test2.json','w', encoding='utf8') as f:
    json.dump(output_tests, f, indent=2, ensure_ascii=False)
