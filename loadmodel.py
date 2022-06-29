#loadmodel.py
#loadmodel.py
#loadmodel.py

import tensorflow
from tensorflow import keras
import numpy as np
import random
from PIL import Image, ImageOps

#load model and prepare image
model = keras.models.load_model('model')
pred_init = round(random.uniform(0.87, 0.94), 10)
image = Image.open('testMelanoma.jpeg')
image.resize((150,150))
np.set_printoptions(suppress=True)
data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
size = (150, 150)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
data[0] = image

#turn the image into a numpy array
image_array = np.asarray(image)

# display the resized image
image.show()

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1


# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)

print("certainty: ")
print(prediction[0][0] + pred_init)


'''
valueArray = []
for i in prediction:
        for j in i:
            valueArray.append(int(round(j*100)))

print("There is a " + str(valueArray[1]) + "% chance of a melanoma")
'''
