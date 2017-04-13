from utility_data_processing_file import Image_Processing
from utility_model_file import MacularHole
from get_bb_coordinate_from_xml import getCoordinatesArray
nb_epoch=10

train_label=getCoordinatesArray('/media/fateh/01D2023161FD29C0/macular_hole/gen_images/single_xml',xmax=2588/30,ymax=1958/30)
train_label/=30
print(train_label)

ip=Image_Processing(nb_rows=1958/30,nb_cols=2588/30,nb_channels=1)
train_data=ip.prep_data('/media/fateh/01D2023161FD29C0/macular_hole/train-one-im.txt',train_file=False)
train_data/=255
#(data_train,label_train),(data_val,label_val)=ip.train_val_split_data(train_data,train_label,nb_classes=4,test_size=0)

#print(label_train)

m1=MacularHole(1958/30,2588/30,1)
model=m1.bounding_box_regression()

model.compile(loss='mse',optimizer='adagrad')
model.fit(train_data,train_label,validation_data=(train_data,train_label),batch_size=1, verbose=1,nb_epoch=nb_epoch)
model.save('regression_approach1.h5')
