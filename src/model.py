from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from datagen import datagen
from PIL import Image
from loss import gradient_importance

target_size = (256, 256)
source_rescale = (56, 56)
batch_size = 32
nb_epoch = 50
# training samples
samples_per_epoch = batch_size
# testing samples
nb_val_samples = batch_size


model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=(*target_size, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))


model.compile(loss=gradient_importance, optimizer='adadelta')


train_generator = datagen('/home/gbentor/OpenUni/DataScience/de-pixalize/data/fr-GYFAt4/small_train', source_rescale, target_size, batch_size)
test_generator = datagen('/home/gbentor/OpenUni/DataScience/de-pixalize/data/fr-GYFAt4/small_test', source_rescale, target_size, batch_size)


# model.fit(train_generator, validation_data=test_generator, batch_size=batch_size, epochs=1)

model.fit(x=train_generator,
                validation_data=test_generator,
                epochs=nb_epoch,
                steps_per_epoch=samples_per_epoch,
                validation_steps=nb_val_samples,
                workers=1,
                callbacks=[TensorBoard(log_dir='/tmp/enhancer', histogram_freq=0, write_graph=False),
                ModelCheckpoint('saved_models/model_laplace.h5', monitor='val_loss', mode='auto')],
                )

