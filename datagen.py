#datagen.py
#datagen.py
#datagen.py

#import pandas
import numpy
#import scikit
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


def performTranslation(filepath, savedir, moleOrMel):
    img = load_img(savedir + '/' + filepath)
    print("filepath is: " + filepath)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=savedir,
                              save_prefix=moleOrMel,
                              save_format='jpeg'):
        i += 1
        if i > 30:
            break


for fname in os.listdir("data/melanoma"):
    if(fname[-5:] == ".jpeg"):
        performTranslation(fname, "data/melanoma", 'melanoma')

for fname in os.listdir("data/mole"):
    if(fname[-5:] == ".jpeg"):
        performTranslation(fname, "data/mole", 'mole')
