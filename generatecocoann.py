import json
import os
import cv2

root_path = 'E:/study/AI/tiny_coco'
phase = 'val'
split = 630
classes = ['bird','car','dog','lizard','turtle']
dataset = {'images': [], 'annotations': [], "categories": []}
train_dataset = {'images': [], 'annotations': [], "categories": []}
val_dataset = {'images': [], 'annotations': [], "categories": []}
for i, cls in enumerate(classes, 1):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'LBC_vid2coco'})

# 读取images文件夹的图片名称
indexes = [f for f in os.listdir(os.path.join(root_path, 'images'))]

cat_cnt = 0
for k, index in enumerate(indexes):
    cat_cnt +=1
    # 用opencv读取图片，得到图像的宽和高
    im = cv2.imread(os.path.join(root_path, 'images/') + index)
    height, width, _ = im.shape
    if cat_cnt <=126:
        train_dataset['images'].append({'file_name': index,
                              'id': k,
                              'width': width,
                              'height': height})
    else:
        val_dataset['images'].append({'file_name': index,
                              'id': k,
                              'width': width,
                              'height': height})
        if cat_cnt == 180:
            cat_cnt = 0

    # 添加图像的信息到dataset中
    dataset['images'].append({'file_name': index,
                              'id': k,
                              'width': width,
                              'height': height})

cnt = 0

for id, cat in enumerate(classes,1):
    cat_cnt = 0
    with open(os.path.join(root_path, cat+'_gt.txt')) as tr:
        annos = tr.readlines()
    for ii, anno in enumerate(annos):
        parts = anno.strip().split()
        parts[0] = parts[0].zfill(6)
        parts[0] = cat+'_'+parts[0]+'.JPEG'
        if parts[0] == indexes[cnt]:
            cls_id = id
            # x_min
            x1 = float(parts[1])
            # y_min
            y1 = float(parts[2])
            # x_max
            x2 = float(parts[3])
            # y_max
            y2 = float(parts[4])
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            cat_cnt +=1
            if cat_cnt <=126:
                train_dataset['annotations'].append({
                    'id': int(len(dataset['annotations']) + 1),                
                    'image_id': cnt,
                    'category_id': int(cls_id),
                    'bbox': [x1, y1, width, height],
                    'area': width * height,            
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
            else:
                val_dataset['annotations'].append({
                    'id': int(len(dataset['annotations']) + 1),                
                    'image_id': cnt,
                    'category_id': int(cls_id),
                    'bbox': [x1, y1, width, height],
                    'area': width * height,            
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
            dataset['annotations'].append({
                'id': int(len(dataset['annotations']) + 1),                
                'image_id': cnt,
                'category_id': int(cls_id),
                'bbox': [x1, y1, width, height],
                'area': width * height,            
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })
            cnt+=1
            if(cat_cnt == 180):
                break


folder = os.path.join(root_path, 'annotations')
print(len(dataset['annotations']))

if not os.path.exists(folder):
  os.makedirs(folder)
train_json_name = os.path.join(root_path, 'annotations/traintinycoco.json')
val_json_name = os.path.join(root_path, 'annotations/valtinycoco.json')
with open(train_json_name, 'w') as f:
  json.dump(train_dataset, f)

with open(val_json_name, 'w') as f:
  json.dump(val_dataset, f)
             

        
