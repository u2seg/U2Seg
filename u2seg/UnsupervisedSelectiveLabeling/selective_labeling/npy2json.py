import numpy as np
import json
import os

# 加载numpy数组
cluster_labels = np.load('/home/xinyang/U2Seg_Project/UnsupervisedSelectiveLabeling/selective_labeling/saved/imagenet_usl_dino_0.2/cluster_labels_300_0.npy')

# 将numpy数组转换为字典
json_data = {}
for i, label in enumerate(cluster_labels):
    img_id = f"{i}.jpg"
    json_data[img_id] = int(label)

# 保存为JSON文件
json_path = 'usl_dino_800_decode.json'
with open(json_path, 'w') as f:
    json.dump(json_data, f)