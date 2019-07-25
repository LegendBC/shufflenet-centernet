import numpy as np
import tensorflow as tf
import cv2
import glob
import os

load_img_path="E:/study/AI/tiny_vid/tiny_vid"
categories = ["bird","car","dog","lizard","turtle"]
cat_dict = {'bird':'0', 'car':'1','dog':'2','lizard':'3','turtle':'4'}
save_img_path="E:/study/AI/tiny_coco"

if not os.path.exists(load_img_path):
    print("图片文件夹不存在...")
else:
    print("开始转换...")
    print("正在转换到：coco...")
cnt = 0
for category in categories:
    print("开始处理",category)
    path_name = load_img_path+'/'+category
    for item in os.listdir(path_name):
        os.rename(os.path.join(path_name,item),os.path.join(path_name,category+'_'+item)) 
        cnt+=1
        print("转换",cnt,"图片")
    