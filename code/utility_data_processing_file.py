from __future__ import print_function
import cv2
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

class Image_Processing:

    def __init__(self,nb_rows=360,nb_cols=480,nb_channels=3):
        self.nb_rows=nb_rows
        self.nb_cols = nb_cols
        self.nb_channels=nb_channels

    def normalized(self,rgb):
        norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

        b = rgb[:, :, 0]
        g = rgb[:, :, 1]
        r = rgb[:, :, 2]

        norm[:, :, 0] = cv2.equalizeHist(b)
        norm[:, :, 1] = cv2.equalizeHist(g)
        norm[:, :, 2] = cv2.equalizeHist(r)
        return norm

    def prep_data(self,path,train_file=True):
        if(train_file is True):
            train_data = []
            train_label = []
            with open(path) as f:
                txt = f.readlines()
                txt = [line.split(',') for line in txt]
            for i in range(len(txt)):
                print(txt[i][0])
                if(self.nb_channels==1):
                    train_data.append(np.resize((cv2.imread(txt[i][0], 0)),
                                                (self.nb_rows, self.nb_cols, self.nb_channels)))
                else:
                    train_data.append(np.resize(self.normalized(cv2.imread(txt[i][0],1)),(self.nb_rows,self.nb_cols,self.nb_channels)))
                train_label.append(int(txt[i][1]))
                print('.', end='')
                print(train_label)
                print(len(train_data))
            train_data=np.asarray(train_data)
            train_label=np.asarray(train_label)
            print (train_data.shape)
            return np.asarray(train_data,dtype=float),np.asarray(train_label,dtype=int)
        else:
            test_data = []
            with open(path) as f:
                txt = f.readlines()
            for i in range(len(txt)):
                if (self.nb_channels == 1):
                    test_data.append(np.resize((cv2.imread(txt[i][0], 0)),
                                                (self.nb_rows, self.nb_cols, self.nb_channels)))
                else:
                    test_data.append(np.resize(self.normalized(cv2.imread(txt[i][0], 1)),
                                                (self.nb_rows, self.nb_cols, self.nb_channels)))
                #test_data.append(self.normalized(cv2.imread(txt[i].strip())))
                print('.', end='')
                print(np.asarray(test_data).shape)
            return np.array(test_data,dtype=float)

    def train_val_split_data(self,data, labels,nb_classes=2,test_size=0.15):
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size,random_state=12)
        labels_train = to_categorical(labels_train, nb_classes)
        labels_test = to_categorical(labels_test, nb_classes)
        return (data_train, labels_train), (data_test, labels_test)

    def dataAugmentation(self,data_train):
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range = 15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        datagen.fit(data_train)
        return datagen