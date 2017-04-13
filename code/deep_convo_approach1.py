from utility_data_processing_file import Image_Processing
from utility_model_file import MacularHole
from keras.optimizers import sgd

nb_classes=2
nb_epoch=500

ip=Image_Processing(nb_rows=1958/2,nb_cols=2588/2,nb_channels=3)
train_data,label_data=ip.prep_data('/media/fateh/01D2023161FD29C0/macular_hole/train-one-im.txt',train_file=True)

train_data/=255
(data_train,label_train),(data_val,label_val)=ip.train_val_split_data(train_data,label_data,nb_classes=2,test_size=0)

#datagen=ip.DataAugmentation(data_train)

m1=MacularHole(nb_rows=1958/2,nb_cols=2588/2,nb_channels=3)

model=m1.DeepConvo(nb_classes)

SGD=sgd(lr=0.1,momentum=0.9)

model.compile(optimizer=SGD,loss='categorical_crossentropy',metrics=['accuracy'])

#model.fit_generator(datagen.flow(data_train, label_train, batch_size=1),
#                    samples_per_epoch=1, nb_epoch=nb_epoch,validation_data=(data_train,label_train))
model.fit(data_train,label_train,validation_data=(data_train,label_train),batch_size=1, verbose=1,nb_epoch=nb_epoch)

model.save('model_approach1.h5')