import numpy as np
import tensorflow as tf
import cv2
import glob
import os

# def create_tfrecords(img_path,txt_path,tfrecords,img_size=384,is_color=True):
#     '''

#     :param img_path: image path
#     :param txt_path: label txt file,  format: imgpath classid1 x1min x1max y1min y1max classid2 x2min x2max y2min y2max ...
#     :param tfrecords:tfrecords name
#     :param img_size:image size
#     :param is_color:image is color or gray
#     :return:None
#     '''
img_size=128
img_path="E:/study/AI/tiny_vid/tiny_vid"
txt_path ="E:/study/AI/tiny_vid/tiny_vid"
train_tfrecords = "E:/study/AI/tiny_vid/tiny_vid/train_vid.tfrecords"
val_tfrecords = "E:/study/AI/tiny_vid/tiny_vid/val_vid.tfrecords"
categories = ["bird","car","dog","lizard","turtle"]
cat_dict = {'bird':'0', 'car':'1','dog':'2','lizard':'3','turtle':'4'}
category_train_num = 18*7
category_val_num = 18*3
#     img_path = "G:/asdsa/JPEGImages"
#     txt_path = "G:/asdsa/voc2012_train_bbox.txt"
#     tfrecords = "G:/asdsa/voc_train_512x512_2.tfrecords"

if not os.path.exists(img_path):
    print("图片文件夹不存在...")
else:
    print("开始转换...")
    print("正在转换到：tfrecords...")
train_writer = tf.python_io.TFRecordWriter(train_tfrecords)
val_writer = tf.python_io.TFRecordWriter(val_tfrecords)
global line
total_img_num = 0
for category in categories:
    print("开始处理",category)
    category_txt_path = txt_path+'/'+category+'_gt'+'.txt'
    with open(category_txt_path) as f:
        line=f.readline().rstrip('\n')
        img_num=0
        while line:
            split_line=line.split(sep=' ')
            img_name=img_path+'/'+category+'/'+split_line[0].zfill(6)+'.JPEG'
            print(img_name)
            label = split_line[:]
            label[0] = cat_dict[category]
            label = np.asarray(label, dtype=np.float32)
            if not os.path.exists(img_name):
                line = f.readline().rstrip('\n')
                continue
            img_raw = cv2.imread(img_name)
            if img_raw is None:
                continue
            height,width=img_raw.shape[0:2]
            print("ori:",height,width)
            max_size=max(height,width)
            if max_size<=img_size:
                    top=(img_size-height)//2
                    bottom=img_size-top-height
                    left=(img_size-width)//2
                    right=img_size-left-width
            else:#max_size>img_size
                if height>=width:
                    scale=img_size/height
                    height=img_size
                    width=int(width*scale)
                    top=0
                    bottom=0
                    left = (img_size - width) // 2
                    right = img_size - left - width
                    label_scale = [1, scale, scale, scale, scale] * (len(label) // 5)
                    label_scale = np.asarray(label_scale,dtype=np.float32)
                    label = label_scale * label
                else:
                    scale=img_size/width
                    width=img_size
                    height=int(height*scale)
                    top = (img_size - height) // 2
                    bottom = img_size - top - height
                    left = 0
                    right = 0
                    label_scale = [1, scale, scale, scale, scale] * (len(label) // 5)
                    label_scale = np.asarray(label_scale,dtype=np.float32)
                    label=label_scale*label
            print(top,bottom,left,right)
            print(height,width)
            img_raw=cv2.resize(img_raw,(width,height))
            img_raw=cv2.copyMakeBorder(img_raw,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
            # for i in range(len(label)):
            #     if i%5==0:
            #         cv2.rectangle(img_raw,(int(label[i+1]),int(label[i+2])),(int(label[i+3]),int(label[i+4])),(0,255,0))
            # print('padd',img_raw.shape)
            offset=[0,left,top,left,top]*(len(label)//5)
            offset = np.asarray(offset, dtype=np.float32)
            label = label+offset
            height=img_size
            width=img_size

            label = label.tobytes()
            img_raw_new = img_raw.tobytes()  # 将图片转化为原生bytes
            # tf.train.Example来定义我们要填入的数据格式，然后使用tf.python_io.TFRecordWriter来写入
            # 一个Example中包含Features，Features里包含Feature（这里没s）的字典。最后，Feature里包含有一个 FloatList，
            # 或者BytesList，或者Int64List
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        # example对象对label和image数据进行封装
                        "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_new])),
                        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))}))

            img_num += 1
            total_img_num +=1
            print("已处理%d幅图..."%(img_num))
            if img_num<=category_train_num:
                print("train:",img_num)
                train_writer.write(example.SerializeToString())  # 序列化为字符串
            else:
                print("val:",img_num)
                val_writer.write(example.SerializeToString())

            if img_num==180:
                break
            line = f.readline().rstrip('\n')
train_writer.close()
val_writer.close()
print("转换结束...")
print("共计",total_img_num,"幅图像...")

# if __name__=='__main__':

#     img_path = "G:/asdsa/JPEGImages"
#     txt_path = "G:/asdsa/voc2012_train_bbox.txt"
#     tfrecords = "G:/asdsa/voc_train_512x512_2.tfrecords"
#     create_tfrecords(img_path, txt_path, tfrecords, img_size=512, is_color=True)
