# coding: utf-8

import json
import zipfile

parent = './'
print(parent+'annotations_goal.json')
# 读入json文件
# with open(parent+'annotations_state_20220508.json', 'r',encoding='utf-8') as f:
with open(parent + 'annotations_goal.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 划分数据，形成train、val、test
print('现有的数据量为：', len(data))

dataname = ['test', 'val', 'train']  # val不是必须的
# dataname = ['test','train']   #val不是必须的

splited_data = {
    "test_data": data[0:60],
    "val_data": data[0:60],
    "train_data": data[60:]
}

for name in dataname:
    print('{}数据集数据量为：{}'.format(name, len(splited_data['{}_data'.format(name)])))

    split_json_file = parent + '{}.json'.format(name)
    with open(split_json_file, 'w', encoding='utf-8') as f:
        json.dump(splited_data['{}_data'.format(name)], f, ensure_ascii=False, sort_keys=True, indent=4)

    f = zipfile.ZipFile(split_json_file + '.zip', 'w', zipfile.ZIP_DEFLATED)  # zipfile.ZIP_STORED不压缩
    f.write(split_json_file, arcname='{}.json'.format(name))
    f.close()
