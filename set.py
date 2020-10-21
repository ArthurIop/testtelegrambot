from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model, model_from_json
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing import image
import numpy as np


import matplotlib.pyplot as plt

image_file_name = 'C:\\Users\\egiazaryan\\Desktop\\a.2.jpg'
img = image.load_img(image_file_name, target_size=(14, 14))
plt.imshow(img)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Загружаем данные об архитектуре сети из файла json
json_file = open("C:\\Users\\egiazaryan\\PycharmProjects\\pythonProject11\\golosovalka.json", "r")


loaded_model_json = json_file.read()


json_file.close()
# Создаем модель на основе загруженных данных
loaded_model = model_from_json(loaded_model_json)
# Загружаем веса в модель
loaded_model.load_weights("C:\\Users\\egiazaryan\\PycharmProjects\\pythonProject11\\golosovalka.h5")


print(np.argmax(loaded_model.predict(img_array)))


