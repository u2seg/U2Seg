import json

# 读取JSON文件
file_path = '/shared/niudt/DATASET/coco/annotations/instances_train2017.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# 打印JSON文件的键
print("Keys in the JSON file:")
for key in data.keys():
    # if key == "images":
    #     import pdb
    #     pdb.set_trace()
    print(key)

# 如果你想要查看每个键下的子键或子数据的结构，可以进一步扩展这个脚本
import json

# 载入JSON文件
file_path = '/shared/niudt/DATASET/coco/annotations/instances_train2017.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# 提取所有的file_name并保存到列表中
file_names = [item['file_name'] for item in data['images']]

# 将列表保存到本地
output_path = '/home/xinyang/Panoptic-CutLER/data/coco_img_file_names.txt'
with open(output_path, 'w') as f:
    for file_name in file_names:
        f.write(file_name + '\n')

print(f"File names have been saved to {output_path}.")
