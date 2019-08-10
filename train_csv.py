import csv
import pandas as pd
from matplotlib import pyplot as plt
from mxnet import gluon,autograd,nd
import mxnet as mx
import os
import numpy as np
from PIL import Image
import random
from gluoncv.data.transforms import experimental

class Reader(gluon.data.Dataset):

    def __init__(self,root,read,transform=None,data_argument=None,k=5):
        self.root=root
        data1=pd.read_csv('./data/train_with_annotations_vinh.csv')
        data2=pd.read_csv('./data/train_with_points.csv')
        t_data=pd.merge(data1,data2,on='Image')
        t_data=t_data.drop(['whaleID_y'], axis=1)
        if data_argument:
            t_data=self.sample_enhance(t_data,5)
        self.t_data_metric=t_data
        t_data=t_data.values
        self.t_data=t_data
        self._k=k
        self._data_argument=data_argument
        self.items=[]
        self._transform=transform
        assert read in ['ssd','theta','metric','test_metric','cls']
        self._read=read
        if read=='ssd':
            self._read_image()
        elif read=='theta':
            self._read_theta()
        elif read=='metric' :
            self._read_metric()
        elif read=='cls':
            self._read_cls()
        self._exts = ['.jpg', '.jpeg', '.png', '.npy']

    def sample_enhance(self,orig_data, add_num=5):
        Id_counts = orig_data['whaleID_x'].value_counts()
        add_num_cal = lambda x: int((1 - min(x, 50) / 50) * add_num)
        enhance_sample = [orig_data]
        for whaleID_x in Id_counts.index:
            num = Id_counts[whaleID_x]
            enhance_num = add_num_cal(num)
            for i in range(enhance_num // num):
                enhance_sample.append(orig_data[orig_data['whaleID_x'] == whaleID_x])
            enhance_sample.append(orig_data[orig_data['whaleID_x'] == whaleID_x].sample(enhance_num % num))
        return pd.concat(enhance_sample)

    def _read_image(self):
        classes_name=self.t_data[:,1]
        self.classes=list(set(classes_name))
        for data in self.t_data:
            abs_path=os.path.join(self.root,data[0])
            labels=[]
            id=0
            x1=int(data[2])
            y1=int(data[3])
            x2=int(data[2]+data[4])
            y2=int(data[3]+data[5])
            label=[x1,y1,x2,y2,id]
            labels.append(label)
            self.items.append((abs_path,labels))

    def _read_theta(self):
        for data in self.t_data:
            abs_path = os.path.join(self.root, data[0])
            x1 = int(data[2])
            y1 = int(data[3])
            x2 = int(data[2] + data[4])
            y2 = int(data[3] + data[5])
            w = float(data[6]-data[8])
            h = float(data[7]-data[9])
            label = [x1,y1,x2,y2,h,w]
            self.items.append((abs_path, label))

    def _read_metric(self):
        self.t_data_metric.sample(frac=1)
        t_data=self.t_data_metric
        t_data_1=t_data.sort_values(by='whaleID_x')
        t_data=t_data.values
        classes_name=self.t_data[:,1]
        self.classes=sorted(list(set(classes_name)))
        convert=dict(zip(self.classes,list(range(len(self.classes)))))
        line=[]
        for i in range(len(t_data)//self._k):
            a=random.random()
            for j in range(self._k):
                line.append(a)
        deta=len(t_data)-len(line)
        b=random.random()
        for j in range(deta):
            line.append(b)
        t_data_1['line']=line
        t_data_1=t_data_1.sort_values(by='line')
        t_data_1=t_data_1.values
        for data in t_data_1:
            abs_path=os.path.join(self.root,data[0])
            labels=[]
            id = convert[data[1]]
            x1 = int(data[2])
            y1 = int(data[3])
            x2 = int(data[2] + data[4])
            y2 = int(data[3] + data[5])
            # if self._data_argument:
            #     cx=(x1+x2)/2
            #     cy=(y1+y2)/2
            #     w=x2-x1
            #     h=y2-y1
            #     w=w*1.2
            #     h=h*1.2
            #     x1=int(cx-w/2)
            #     y1=int(cy-h/2)
            #     x2=int(cx+w/2)
            #     y2=int(cy+h/2)
            w1=int(data[6]-data[8])
            h1=int(data[7]-data[9])
            if w1 < 0.000001:
                w1 += 1e-6
            if h1 < 0.000001:
                h1 += 1e-6
            label=[x1,y1,x2,y2,w1,h1,id]
            labels.append(label)
            self.items.append((abs_path,labels))

    def _scale(self,scale,label):
        x1,y1,x2,y2=label[:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        w = w * scale
        h = h * scale
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        label[:4]=[x1,y1,x2,y2]
        return label


    def _read_cls(self):
        t_data=self.t_data
        classes_name = self.t_data[:, 1]
        self.classes = sorted(list(set(classes_name)))
        convert = dict(zip(self.classes, list(range(len(self.classes)))))
        for data in t_data:
            abs_path=os.path.join(self.root,data[0])
            labels=[]
            id = convert[data[1]]
            x1 = int(data[2])
            y1 = int(data[3])
            x2 = int(data[2] + data[4])
            y2 = int(data[3] + data[5])
            # if self._data_argument:
            #     cx=(x1+x2)/2
            #     cy=(y1+y2)/2
            #     w=x2-x1
            #     h=y2-y1
            #     w=w*1.1
            #     h=h*1.1
            #     x1=int(cx-w/2)
            #     y1=int(cy-h/2)
            #     x2=int(cx+w/2)
            #     y2=int(cy+h/2)
            w1=int(data[6]-data[8])
            h1=int(data[7]-data[9])
            if w1 < 0.000001:
                w1 += 1e-6
            if h1 < 0.000001:
                h1 += 1e-6
            label=[x1,y1,x2,y2,w1,h1,id]
            labels.append(label)
            self.items.append((abs_path,labels))
            #此处可放大box再append一次
            # if self._data_argument:
            #     label=self._scale(1.2,label)
            #     self.items.append((abs_path, [label]))
            #     label=self._scale(1/1.2,label)
            #     self.items.append((abs_path, [label]))





    def __getitem__(self,id):
        img_path=self.items[id][0]
        img=mx.image.imread(img_path)
        label=self.items[id][1]
        if self._read == 'metric' or self._read=='cls':
            #关于角度的数据增强
            img_shape = img.shape
            x1, y1, x2, y2 = label[0][:4]
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > img_shape[1]:
                x2 = img_shape[1]
            if y2 > img_shape[0]:
                y2 = img_shape[0]
            w = x2 - x1
            h = y2 - y1
            w1,h1=label[0][4:6]
            theta=np.arctan(w1/h1)
            deg = np.rad2deg(theta)
            if self._data_argument:
                rate=random.randint(-8,9)
                deg+=rate
            img = mx.image.fixed_crop(img,x1,y1,w,h,size=(256,256))
            img = img.asnumpy()
            img= Image.fromarray(img)
            if h1>0:
                img = img.rotate(-90-deg)
            elif h1<0:
                img = img.rotate(90-deg)
            img = np.array(img)
            # plt.imshow(img)
            # plt.show()
            img=nd.array(img)
            label=[[label[0][-1]]]

        if self._read=='theta':
            img_shape=img.shape
            x1, y1, x2, y2 = label[:4]
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > img_shape[1]:
                x2 = img_shape[1]
            if y2 > img_shape[0]:
                y2 = img_shape[0]
            w = x2 - x1
            h = y2 - y1
            if w>=h:
                deta_h=w-h
                y1-=deta_h/2
                y1=int(y1)
                if y1<0:
                    y1=0
                h=w
            else:
                deta_w=h-w
                x1-=deta_w/2
                x1=int(x1)
                if x1<0:
                    x1=0
                w=h
            if x1+w>img_shape[1]:
                x1=img_shape[1]-w
            if y1+h>img_shape[0]:
                y1=img_shape[0]-h
            img = mx.image.fixed_crop(img, x1, y1, w, h,size=(512,512))
            img=img.asnumpy()
            img = Image.fromarray(img)
            img=np.array(img)
            # plt.imshow(img)
            # plt.show()

            label=label[4:]
            label=[float(label[0]/100),float(label[1]/100)]

        if self._transform is not None:
            return self._transform(img, np.array(label))

        # if self._read=='cls':
        #     img = experimental.image.random_color_distort(img)
        return img, np.array(label)

    def __len__(self):
        return len(self.items)

class Test_reader(gluon.data.Dataset):
    def __init__(self,root,csv_path,read,transform=None):
        self._root=root
        self._read=read
        data=pd.read_csv(csv_path)
        self.t_data=data.values
        self._transform=transform
        self.items = []
        if self._read=='theta':
            self._read_image()
        if self._read=='pred_box':
            self._read_predict_box()
        if self._read=='predict_cls':
            self._read_cls()


    def _read_image(self):
        for data in self.t_data:
            abs_path=os.path.join(self._root,data[1])
            box=data[2:]
            self.items.append((abs_path,box))

    def _read_predict_box(self):
        for data in self.t_data:
            abs_path = os.path.join(self._root, data[0])
            self.items.append(abs_path)

    def _read_cls(self):
        t_data=self.t_data[:,2:]
        for data in t_data:
            abs_path=os.path.join(self._root, data[0])
            label=data[1:]
            self.items.append((abs_path,label))

    def __getitem__(self, id):
        if self._read=='theta':
            img=mx.image.imread(self.items[id][0])
            img_shape = img.shape
            x1,y1,x2,y2=self.items[id][1]
            if x1<0:
                x1=0
            if y1<0:
                y1=0
            if x2>img_shape[1]:
                x2=img_shape[1]
            if y2>img_shape[0]:
                y2=img_shape[0]
            w = x2 - x1
            h = y2 - y1
            if w >= h:
                deta_h = w - h
                y1 -= deta_h / 2

                if y1 < 0:
                    y1 = 0
                h = w
            else:
                deta_w = h - w
                x1 -= deta_w / 2

                if x1 < 0:
                    x1 = 0
                w = h
            if x1 + w > img_shape[1]:
                # w = img_shape[1] - x1
                x1 = img_shape[1] - w
            if y1 + h > img_shape[0]:
                # h = img_shape[0] - y1
                y1 = img_shape[0] - h
            try:
                img=mx.image.fixed_crop(img,int(x1),int(y1),int(w),int(h),size=(512,512))
            except:
                print(self.items[id][0])

        if self._read=='predict_cls':
            img = mx.image.imread(self.items[id][0])
            img_shape = img.shape
            x1, y1, x2, y2 = self.items[id][1][:4]
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > img_shape[1]:
                x2 = img_shape[1]
            if y2 > img_shape[0]:
                y2 = img_shape[0]
            w = x2 - x1
            h = y2 - y1

            try:
                img = mx.image.fixed_crop(img, int(x1), int(y1), int(w), int(h), size=(256, 256))
            except:
                print(self.items[id][0])
            img=img.asnumpy()
            img=Image.fromarray(img)
            deta_h,deta_w=self.items[id][1][4:]
            theta=np.arctan(deta_h/deta_w)
            deg=np.rad2deg(theta)
            if deta_w>=0:
                img=img.rotate(-90-deg)
            else:
                img=img.rotate(90-deg)

            img=np.array(img)
            # plt.imshow(img)
            # plt.show()
            img=nd.array(img)

        if self._read=='pred_box':
            img=mx.image.imread(self.items[id])
            h=img.shape[0]
            w=img.shape[1]
            return img,nd.array((h,w))
        return img



    def __len__(self):
        return len(self.items)

#a=Reader('./data/imgs',read='theta')


