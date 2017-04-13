from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model

class MacularHole:
    def __init__(self,nb_rows=360,nb_cols=480,nb_channels=3):
        self.nb_rows=nb_rows
        self.nb_cols = nb_cols
        self.nb_channels=nb_channels

    def deepConvo_Classification(self,nb_classes):
        model=Sequential()
        model.add(Convolution2D(64,3,3,input_shape=(self.nb_rows,self.nb_cols,self.nb_channels),border_mode='same',activation='relu'))
        model.add(Convolution2D(64,3,3,border_mode='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(128,3,3,border_mode='same',activation='relu'))
        model.add(Convolution2D(128,3,3,border_mode='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(64,3,3,border_mode='same',activation='relu'))
        model.add(Convolution2D(64,3,3,border_mode='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128))
        #model.add(Dense(64))
        #model.add(Dense(16))
        model.add(Dense(nb_classes,activation='softmax'))
        return model

    def bounding_box_regression(self):
        inputs = Input(shape=(self.nb_rows,self.nb_cols, self.nb_channels))
        x = Convolution2D(64,3,3,border_mode='same')(inputs)
        x = Convolution2D(64,3,3,border_mode='same')(x)
        x = Convolution2D(64,3,3,border_mode='same')(x)
        x=Flatten()(x)
        x = Dense(128)(x)
        x=Dense(4,activation='linear')(x)
        model = Model(input=inputs, output=x)
        return model
