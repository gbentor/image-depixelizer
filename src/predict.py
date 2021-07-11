from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from datagen import datagen
from PIL import Image
from loss import gradient_importance

# SETTINGS
target_size = (256, 256)
source_rescale = (56, 56)
batch_size = 32
nb_epoch = 10
# training samples
samples_per_epoch = 50
# testing samples
nb_val_samples = 10


train_generator = datagen('/home/gbentor/OpenUni/DataScience/de-pixalize/data/fr-GYFAt4/small_train', source_rescale, target_size, batch_size)
test_generator = datagen('/home/gbentor/OpenUni/DataScience/de-pixalize/data/fr-GYFAt4/small_test', source_rescale, target_size, batch_size)


model = load_model('saved_models/model_laplace.h5', custom_objects={'gradient_importance': gradient_importance})

X,y = next(test_generator)
for i in range(batch_size):
    predicted = model.predict(X[i].reshape(-1, *(X[i]).shape))
    p_im = Image.fromarray(predicted[0].reshape(*target_size)*255).convert('RGB')
    X_im = Image.fromarray(X[i].reshape(*target_size)*255).convert('RGB')
    y_im = Image.fromarray(y[i].reshape(*target_size)*255).convert('RGB')
    p_im.save('images/p_{}.png'.format(i+1))
    X_im.save('images/X_{}.png'.format(i+1))
    y_im.save('images/y_{}.png'.format(i+1))
    X, y = next(test_generator)