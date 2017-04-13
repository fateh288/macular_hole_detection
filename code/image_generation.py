from keras.preprocessing.image import ImageDataGenerator
from utility_data_processing_file import Image_Processing
from matplotlib import pyplot
import os
import cv2

images=Image_Processing(1958,2588,1)
train_data,label_data=images.prep_data('/media/fateh/01D2023161FD29C0/macular_hole/train_2.txt',train_file=True)
datagen=images.dataAugmentation(train_data)
#os.makedirs('../gen_images')
#datagen.flow(train_data,label_data, batch_size=10, save_to_dir='gen_images', save_prefix='gen_', save_format='jpg')

for X_batch, y_batch in datagen.flow(train_data,label_data, batch_size=10, save_to_dir='../gen_images', save_prefix='gen_', save_format='png'):
	# create a grid of 3x3 images
	for i in range(0, 10):
		#pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i].reshape(1958,2588))
		#pyplot.imshow(X_batch[i].reshape(1958,2588))
		cv2.waitKey()
	# show the plot
	pyplot.show()
	break