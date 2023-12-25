# 从本地文件读取file_names列表
output_path = '/home/xinyang/Panoptic-CutLER/data/coco_img_file_names.txt'
with open(output_path, 'r') as f:
    file_names = [line.strip() for line in f.readlines()]

# 对file_names进行排序
file_names.sort()

print(file_names[:10])  # 打印排序后的前10个file_name