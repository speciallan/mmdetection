#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import cv2
import json
import numpy as np
import os
from xml.etree import ElementTree
import matplotlib.pyplot as plt
import random
from shutil import copyfile
import os
import xml.etree.cElementTree as ET
from xml.dom import minidom

#保存xml格式的
def save_annotations(filepath, filename, image, boxes):
    #保存图像
    cv2.imwrite(os.path.join(os.path.join(filepath, "JPEGImages"), filename + ".jpg"), image)

    #保存画框图像
    for i in range(len(boxes)):
        cv2.rectangle(image,
                      (boxes[i]['xmin'],#xmin
                       boxes[i]['ymin']), #ymin
                      (boxes[i]['xmax'],#xmax
                       boxes[i]['ymax']), #ymax
                      (0,255,0),
                      1)
    cv2.imwrite(os.path.join(os.path.join(filepath, "DrawBoxes"), filename + ".jpg"), image)

    #保存labels
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = filename + ".jpg"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image.shape[0])
    ET.SubElement(size, "height").text = str(image.shape[1])
    ET.SubElement(size, "depth").text = str(image.shape[2])
    for box in boxes:
        object = ET.SubElement(root, "object")
        ET.SubElement(object, "name").text = box["name"]
        ET.SubElement(object, "pose").text = "Unspecified"
        ET.SubElement(object, "truncated").text = "0"
        ET.SubElement(object, "difficult").text = "0"
        bndbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(box["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(box["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(box["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(box["ymax"])

    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    myfile = open(os.path.join(os.path.join(filepath, "Annotations"), filename + ".xml"), "w")
    myfile.write(xmlstr)

#获得全部的图片路径
def get_all_img_path(original_path):
    img_path=[]
    annotation_path=[]
    for files in os.listdir(original_path):
        img_path.append(original_path+files+"/"+files+".jpg")
        annotation_path.append(original_path+files+"/"+"annotation_ship.xlsx")
    return [img_path,annotation_path]

#获得label文件
def get_label(path):
    #获得xls文件
    workbook=xlrd.open_workbook(path)
    worksheet=workbook.sheet_by_name("Work")

    data=[]
    for i in range(1,worksheet.nrows,1):#获取该表总行数
        one_data=[]
        for j in range(1,5,1):#列
            one_data.append(int(worksheet.cell_value(i,j)))
        data.append(one_data)

    return data#返回xmin ymin xmax ymax


#制作目标检测数据集
def make_datasets(all_img_path,annotation_path,save_path,size,overlap):
    #新建需要保存的文件夹
    if os.path.exists(save_path)==False:os.makedirs(save_path)
    if os.path.exists(save_path+"JPEGImages")==False:os.makedirs(save_path+"JPEGImages")
    if os.path.exists(save_path+"Annotations")==False:os.makedirs(save_path+"Annotations")
    if os.path.exists(save_path+"DrawBoxes")==False:os.makedirs(save_path+"DrawBoxes")
    #对每一张图像制作数据集
    for i in range(7,9,1):
        all_get_box=0#累计所有能得到的box
        #获得label
        labels=get_label(annotation_path[i])
        #读入图像
        print("读入"+all_img_path[i])
        img=cv2.imread(all_img_path[i])

        #按步长依次制作
        for y in range(0,img.shape[0]-size,size-overlap):#
            for x in range(0,img.shape[1]-size,size-overlap):
                #针对左上角为x、y的图像进行labels分配
                boxes=[]
                for t in range(len(labels)):
                    #对label进行大小值转换
                    the_xmin=-1
                    the_xmax=-1
                    the_ymin=-1
                    the_ymax=-1
                    #进行大小写整合
                    if labels[t][0]<labels[t][2]:
                        the_xmin=labels[t][0]
                        the_xmax=labels[t][2]
                    elif labels[t][2]<labels[t][0]:
                        the_xmin=labels[t][2]
                        the_xmax=labels[t][0]
                    if labels[t][1]<labels[t][3]:
                        the_ymin=labels[t][1]
                        the_ymax=labels[t][3]
                    elif labels[t][3]<labels[t][1]:
                        the_ymin=labels[t][3]
                        the_ymax=labels[t][1]
                    #判断是否在内部
                    if(the_xmin>=x and
                            the_ymin>=y and
                            the_xmax<=x+size and
                            the_ymax<=y+size and #在这个框内
                            the_xmin!=-1 and
                            the_ymin!=-1 and
                            the_xmax!=-1 and
                            the_ymax!=-1):#满足严格大小
                        label_temp=dict([('xmin',the_xmin-x),
                                         ('ymin',the_ymin-y),
                                         ('xmax',the_xmax-x),
                                         ('ymax',the_ymax-y),
                                         ('name','ship')])
                        boxes.append(label_temp)
                #保存有label的图像
                if len(boxes)!=0:
                    #统计完成保存
                    this_img=img[y:y+size,x:x+size,:]
                    save_annotations(save_path,
                                     all_img_path[i][17:19]+"_"+str(y)+"_"+str(x),
                                     this_img, boxes)

                    #统计相关数据并输出
                    all_get_box=all_get_box+len(boxes)
                    print(str(all_img_path[i][17:19])+" "+str(y)+"_"+str(x)+" box:"+str(all_get_box)+"/"+str(len(labels)))

    #从这里开始
    return True

#统计相关的分布
def show_statistics(all_img_path,annotations_path):
    all_w=[]#所有的宽度
    all_h=[]#所有的高度
    sum_label=0#所有的label个数
    one_line=0#统计只有一根线的个数
    w_h_ratio=[]#高宽比

    #开始统计
    for i in range(0,len(annotations_path),1):
        print(annotations_path[i])
        print("读入图像:"+all_img_path[i])
        #        img=cv2.imread(all_img_path[i])#获得图像
        boxes=get_label(annotations_path[i])#获得label
        sum_label=sum_label+len(boxes)#所有label个数累计求和
        for j in range(len(boxes)):
            xmin=0
            ymin=0
            w=0
            h=0
            #计算高宽
            if boxes[j][2]-boxes[j][0]>0:
                xmin=boxes[j][0]
                w=boxes[j][2]-boxes[j][0]
            else:
                xmin=boxes[j][2]
                w=boxes[j][0]-boxes[j][2]
            if boxes[j][3]-boxes[j][1]>0:
                ymin=boxes[j][1]
                h=boxes[j][3]-boxes[j][1]
            else:
                ymin=boxes[j][3]
                h=boxes[j][1]-boxes[j][3]
            #统计
            all_w.append(w)
            all_h.append(h)
            if h!=0:
                w_h_ratio.append(w/h)#比例
            #一根线
            if(w==0 or h==0):
                one_line=one_line+1


            #根据条件，把其固定要求的画框显示下来
    #            if os.path.exists("temp/")==False:os.makedirs("temp/")
    #            if (w==1 or h==1):
    #                overlap=100
    #                cv2.rectangle(img,(xmin,ymin),(xmin+w,ymin+h),(0,255,0),1)
    #                this_img=img[ymin-overlap:ymin+h+overlap,
    #                             xmin-overlap:xmin+w+overlap,:]
    #                cv2.imwrite("temp/"+str(all_img_path[i][17:19])+" "+str(j)+".jpg",this_img)


    #画出统计图
    def draw_data(data,bins,title):
        all_w=np.array(data)
        plt.title(title)
        plt.hist(all_w,bins=bins,color='green',edgecolor='black',histtype='bar')
        plt.grid(linestyle='--', color="gray")
        plt.show()

    #画图
    draw_data(all_w,50,"all_w")
    draw_data(all_h,50,"all_h")
    draw_data(w_h_ratio,50,"w_h_ratio")
    print("one line num: "+str(one_line))
    print("all label "+str(sum_label))

    return all_w




#-------------------------------上次的---------------------------------








#获得全部文件名list
def get_all_file_name(path):
    name=[]
    for files in os.listdir(path):
        files=files.replace(".jpg","")
        name.append(files)
    return name

#输入label.xml的连接输出boxes的位置
#def get_label(path):
#    tree = ElementTree.parse(path)
#    root = tree.getroot()
#    bounding_boxes = []
#    for object_tree in root.findall('object'):
#        class_name = object_tree.find('name').text
#        for bounding_box in object_tree.iter('bndbox'):
#            # 得到归一化坐标
#            xmin = int(bounding_box.find('xmin').text)
#            ymin = int(bounding_box.find('ymin').text)
#            xmax = int(bounding_box.find('xmax').text)
#            ymax = int(bounding_box.find('ymax').text)
#        bounding_box ={'class':class_name,'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax}
#        bounding_boxes.append(bounding_box)
#    return bounding_boxes



#制作分类数据集
def make_classify_datasets(classify_path,img_path,annotations_path,max_size):
    #建立目录
    if os.path.exists(classify_path+"pos/")==False:os.makedirs(classify_path+"pos/")
    if os.path.exists(classify_path+"neg/")==False:os.makedirs(classify_path+"neg/")

    #首先获得所有原始图像的名字
    pos_name_list=get_all_file_name(classify_path+"/from")

    #统计label分布
    show_statistics(pos_name_list,annotations_path)

    #对每一张图像进行处理
    num_use=0#能使用的数量
    for i in range(len(pos_name_list)):
        print(str(i)+"/"+str(len(pos_name_list)))
        boxes=get_label(annotations_path+pos_name_list[i]+".xml")#getbox
        img=cv2.imread(img_path+pos_name_list[i]+".jpg")
        for j in range(len(boxes)):
            #满足要求就统计
            this_w=boxes[j]['xmax']-boxes[j]['xmin']
            this_h=boxes[j]['ymax']-boxes[j]['ymin']
            new_x=int(boxes[j]['xmin']+this_w/2)
            new_y=int(boxes[j]['ymin']+this_h/2)
            if(this_w<=max_size and#宽度小于max_size
                    this_h<=max_size and#高度小于max_size
                    int(new_x-max_size/2)>=0 and#左上角在范围内
                    int(new_y-max_size/2)>=0 and#左上角在范围内
                    int(new_x+max_size/2)<=img.shape[0] and#右下角在范围内
                    int(new_y+max_size/2)<=img.shape[1]#右下角在范围内
            ):
                num_use=num_use+1
                print(num_use)
                #正例保存
                pos_img=img[int(new_y-max_size/2):int(new_y+max_size/2),
                        int(new_x-max_size/2):int(new_x+max_size/2),:]
                cv2.imwrite(classify_path+"pos/"+pos_name_list[i]+"_"+str(j)+".jpg",pos_img)

                #随机选择一个区域负例
                can_get_neg=False
                wait_time=100000
                while can_get_neg==False and wait_time>=0:#循环生成
                    neg_x=int(random.random()*10000)%(img.shape[1]-max_size)
                    neg_y=int(random.random()*10000)%(img.shape[1]-max_size)
                    #判断这个是否满足要求
                    can_use=True
                    for tt in range(len(boxes)):
                        if (neg_x>=boxes[tt]['xmin']-max_size and
                                neg_y>=boxes[tt]['ymin']-max_size and
                                new_x<=boxes[tt]['xmax'] and
                                new_y<=boxes[tt]['ymax']):
                            can_use=False
                            break
                    #满足就保存，不满足就继续
                    if can_use==True:
                        neg_img=img[neg_y:neg_y+max_size,
                                neg_x:neg_x+max_size,:]
                        cv2.imwrite(classify_path+"neg/"+pos_name_list[i]+"_"+str(j)+".jpg",neg_img)
                        can_get_neg=True
                    wait_time=wait_time-1

#训练模型
def train_model(classify_path,num_max,max_size):
    pos_img=np.ones((num_max,max_size,max_size,3))
    neg_img=np.ones((num_max,max_size,max_size,3))
    i=0
    for img in os.listdir(classify_path+"pos"):
        print("read pos "+str(i)+"/"+str(num_max))
        pos_img[i]=cv2.imread(classify_path+"pos/"+img)
        i=i+1
    i=0
    for img in os.listdir(classify_path+"neg"):
        print("read neg "+str(i)+"/"+str(num_max))
        neg_img[i]=cv2.imread(classify_path+"neg/"+img)
        i=i+1

    img_data=np.concatenate((pos_img,neg_img))
    #数据预处理
    img_data=img_data.astype('float32')/255

    #制作标签
    labels=np.concatenate((np.ones((num_max,)),np.zeros((num_max,))))


    #打乱训练集数据
    indices=np.arange(img_data.shape[0])
    np.random.shuffle(indices)
    img_data= img_data[indices]
    labels= labels[indices]


    #搭建网络框架
    model=models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(max_size,max_size,3)))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='sigmoid'))
    model.summary()


    #编译
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #拟合

    model.fit(img_data[0:int(2*num_max*0.7)],
              labels[0:int(2*num_max*0.7)],
              epochs=15,
              batch_size=16,
              validation_data=(img_data[int(2*num_max*0.7):2*num_max],labels[int(2*num_max*0.7):2*num_max]))

    #保存模型
    model.save("model.h5")

#错误分析
def wrong_analyse(classify_path,max_size):
    #建立目录
    if os.path.exists(classify_path+"wrong/")==False:os.makedirs(classify_path+"wrong/")

    model=models.load_model("model.h5")
    model.summary()

    if os.path.exists(classify_path+"wrong/is0but1")==False:os.makedirs(classify_path+"wrong/is0but1")
    if os.path.exists(classify_path+"wrong/is1but0")==False:os.makedirs(classify_path+"wrong/is1but0")
    #正例负例依次预测
    sum_wrong=0
    for img in os.listdir(classify_path+"pos"):
        pos_img=cv2.imread(classify_path+"pos/"+img)
        pos_img_use=np.reshape(pos_img,(1,max_size,max_size,3))/255
        if model.predict(pos_img_use)[0][0]<0.5:
            print("true 1 predict "+str(model.predict(pos_img_use)[0][0]))
            sum_wrong=sum_wrong+1
            cv2.imwrite(classify_path+"wrong/is1but0/"+img,pos_img)

    for img in os.listdir(classify_path+"neg"):
        neg_img=cv2.imread(classify_path+"neg/"+img)
        neg_img_use=np.reshape(neg_img,(1,max_size,max_size,3))/255
        if model.predict(neg_img_use)[0][0]>0.5:
            print("true 0 predict "+str(model.predict(neg_img_use)[0][0]))
            sum_wrong=sum_wrong+1
            cv2.imwrite(classify_path+"wrong/is0but1/"+img,neg_img)
    print("sum wrong = "+str(sum_wrong))

#滑动窗口预测
def win_predict(img_path,annotations_path,pre_name,max_size,threshold):
    model=models.load_model("model.h5")
    model.summary()
    #读入labels
    boxes=get_label(annotations_path+pre_name+".xml")
    #读入图像
    img=cv2.imread(img_path+pre_name+".jpg")
    draw_img=cv2.imread(img_path+pre_name+".jpg")
    #循环窗口生成
    for x in range(0,img.shape[0]-max_size,int(max_size/8)):
        for y in range(0,img.shape[0]-max_size,int(max_size/8)):
            #对该模块进行预测
            this_img=np.reshape(img[y:y+max_size,x:x+max_size,:],(1,max_size,max_size,3))/255
            this_label=model.predict(this_img)[0][0]
            print(str(y)+":"+str(x)+" confidence:"+str(this_label))

            if this_label>=threshold:
                print(this_label)
                #画框
                cv2.rectangle(draw_img,(x,y),(x+max_size,y+max_size),(0,255,255),1)
                #添加文本
                cv2.putText(draw_img,format(this_label,'.4f'),(x,y), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

    #画出真正的labels框
    for j in range(len(boxes)):
        #画框
        cv2.rectangle(draw_img,
                      (boxes[j]['xmin'],boxes[j]['ymin']),
                      (boxes[j]['xmax'],boxes[j]['ymax']),
                      (255,255,0),
                      1)

    #保存输出
    cv2.imwrite("final.jpg",draw_img)


#按要求制作不同的数据集
def make_different_datasets(pos_name_list):
    all_w=[]#统计全部的w
    all_h=[]#统计全部的h
    boxes_of_one_img=[]#统计每张图像的boxes数量
    white_area=[]#每张图二值化之后的白色区域做统计
    connected=[]#每张图的连通域统计
    for i in range(len(pos_name_list)):
        print(str(i)+"/"+str(len(pos_name_list)))
        boxes=get_label(annotations_path+pos_name_list[i]+".xml")#读入labels
        img=cv2.imread(img_path+pos_name_list[i]+".jpg")#读入图像

        boxes_of_one_img.append(len(boxes))#统计boxes数目
        _,binary = cv2.threshold(img[:,:,0],127,255 ,cv2.THRESH_BINARY)#二值化
        kernel = np.ones((5, 5), np.uint8)#膨胀核
        after_dilation = cv2.dilate(binary, kernel)#膨胀操作
        white_area.append((np.sum((binary/255)/3)))#统计二值图像中1面积
        connected_num,_ = cv2.connectedComponents(after_dilation,connectivity=8)#统计连通域个数
        connected.append(connected_num-1)

        for j in range(len(boxes)):
            all_w.append(boxes[j]['xmax']-boxes[j]['xmin'])#统计w
            all_h.append(boxes[j]['ymax']-boxes[j]['ymin'])#统计h




        #根据条件保存数据集
        save_datasets_path="new_datasets/over_3boxes/"
        if os.path.exists(save_datasets_path)==False:os.makedirs(save_datasets_path)
        if os.path.exists(save_datasets_path+"Annotations/")==False:
            os.makedirs(save_datasets_path+"Annotations/")
        if os.path.exists(save_datasets_path+"JPEGImages/")==False:
            os.makedirs(save_datasets_path+"JPEGImages/")
        if os.path.exists(save_datasets_path+"draw_boxes/")==False:
            os.makedirs(save_datasets_path+"draw_boxes/")

        if boxes_of_one_img[i]>=3:
            #保存xml文件
            copyfile(annotations_path+pos_name_list[i]+".xml",
                     save_datasets_path+"Annotations/"+pos_name_list[i]+".xml")#复制
            #图像保存
            copyfile(img_path+pos_name_list[i]+".jpg",
                     save_datasets_path+"JPEGImages/"+pos_name_list[i]+".jpg")#复制
            #画框保存
            for j in range(len(boxes)):
                cv2.rectangle(img,
                              (boxes[j]['xmin'],boxes[j]['ymin']),
                              (boxes[j]['xmax'],boxes[j]['ymax']),
                              (0,255,255,3))
            cv2.imwrite(save_datasets_path+"draw_boxes/"+pos_name_list[i]+".jpg",img)


    #--------画出统计图
    def draw_data(data,bins,title):
        all_w=np.array(data)
        plt.title(title)
        plt.hist(all_w,bins=bins,color='green',edgecolor='black',histtype='bar')
        plt.grid(linestyle='--', color="gray")
        plt.show()

    draw_data(all_w,50,"all_w")
    draw_data(all_h,50,"all_h")
    draw_data(boxes_of_one_img,14,"boxes_of_one_img")
    draw_data(white_area,50,"white_area")

    return connected
