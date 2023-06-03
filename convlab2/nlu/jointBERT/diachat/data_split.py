# 导入模块
import json
import zipfile

parent = '../../../../data/diachat/'
# 读入json文件
with open(parent+'annotations_20220328.json', 'r',encoding='utf-8') as f:
    data = json.load(f)

# 划分数据，形成train、val、test
print('现有的数据量为：', len(data))

test_data = data[0:60]
val_data = data[60:120]
train_data = data[120:]

print('训练集数据量为：', len(train_data))
print('验证集数据量为：', len(val_data))
print('测试集数据量为：', len(test_data))

# 保存文件

train_filename = 'train.json'
train_filepath = parent + train_filename
val_filename = 'val.json'
val_filepath = parent + val_filename
test_filename = 'test.json'
test_filepath = parent + test_filename

with open(train_filepath, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, sort_keys=True, indent=4)


with open(val_filepath, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, sort_keys=True, indent=4)


with open(test_filepath, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, sort_keys=True, indent=4)


f = zipfile.ZipFile(train_filepath+'.zip','w',zipfile.ZIP_DEFLATED)#zipfile.ZIP_STORED不压缩
f.write(train_filepath, arcname=train_filename)
f.close()

f = zipfile.ZipFile(val_filepath+'.zip','w',zipfile.ZIP_DEFLATED)
f.write(val_filepath,arcname=val_filename)
f.close()

f = zipfile.ZipFile(test_filepath+'.zip','w',zipfile.ZIP_DEFLATED)
f.write(test_filepath,arcname=test_filename)
f.close()