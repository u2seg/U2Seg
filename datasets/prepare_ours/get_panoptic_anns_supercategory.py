import json
import os


if __name__ == '__main__':
    for cluster_num in [300, 800]:
        standard_file = json.load(open('../datasets/panoptic_anns/panoptic_val2017.json'))
        map = {92: 1, 93: 1, 95: 2, 100: 3, 107: 4, 109: 1, 112: 4, 118: 5, 119: 6, 122: 7, 125: 8, 128: 2, 130: 4,
                       133: 4, 138: 9, 141: 1, 144: 8, 145: 8, 147: 8, 148: 10, 149: 8, 151: 2, 154: 8, 155: 10, 156: 4, 159: 8,
                       161: 4, 166: 2, 168: 1, 171: 11, 175: 11, 176: 11, 177: 11, 178: 10, 180: 12, 181: 12, 184: 6, 185: 9,
                       186: 13, 187: 14, 188: 4, 189: 4, 190: 5, 191: 8, 192: 15, 193: 6, 194: 8, 195: 3, 196: 7, 197: 2,
                       198: 15, 199: 11, 200: 1}

        for ann in standard_file['annotations']:
            segments_info = ann['segments_info']
            for seg in segments_info:
                s = 1
                if seg['category_id'] in map:
                    seg['category_id'] = map[seg['category_id']] + cluster_num
                    s = 1


        categories = standard_file['categories']

        for cate in categories:
            if cate['id'] in map:
                cate['id'] = map[cate['id']] + cluster_num

        print(categories)
        json_save_path = os.path.join('../datasets/panoptic_anns/panoptic_val2017_{}super.json'.format(cluster_num))
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(standard_file, f, ensure_ascii=False)