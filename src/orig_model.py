from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Reshape,Flatten
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
# tensorboard --logdir=/tmp/

import numpy as np

from datagen import datagen
from loss import gradient_importance

# SETTINGS
target_size = (256, 256)
source_rescale = (128, 128)
batch_size = 32
nb_epoch = 50
# training samples
samples_per_epoch = 290496/batch_size
# testing samples
nb_val_samples = 3020/batch_size

### THE MODEL ####

# input_img = Input(shape=(*target_size, 3))

x = Convolution2D(256, (3, 3), activation='relu', padding='same', input_shape=(*target_size,3))#(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Convolution2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Convolution2D(512, (3, 3), activation='relu', padding='same')(x)
x = Convolution2D(512, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Convolution2D(256,( 3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Convolution2D(1, (3, 3), activation='sigmoid', padding='same')(x)
# x = Dense(units=2, activation='softmax')(x)

model = Model(input_img, x)
model.compile(optimizer='adadelta', loss=gradient_importance)


### (end) THE MODEL ####

### THE DATA ####

train_generator = datagen('/home/gbentor/OpenUni/DataScience/de-pixalize/data/fr-GYFAt4/train', source_rescale, target_size, batch_size)
test_generator = datagen('/home/gbentor/OpenUni/DataScience/de-pixalize/data/fr-GYFAt4/test', source_rescale, target_size, batch_size)


model.fit(x=train_generator,
                validation_data=test_generator,
                epochs=nb_epoch,
                steps_per_epoch=samples_per_epoch,
                validation_steps=nb_val_samples,
                workers=1,
                callbacks=[TensorBoard(log_dir='/tmp/enhancer', histogram_freq=0, write_graph=False),
                ModelCheckpoint('saved_models/model_laplace.h5', monitor='val_loss', mode='auto')],
                )
