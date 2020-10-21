from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense


# Каталог с данными для обучения
train_dir = 'C:\\Users\\egiazaryan\\Desktop\\dataset2\\train'
# Каталог с данными для проверки
val_dir = 'C:\\Users\\egiazaryan\\Desktop\\dataset2\\val'
# Каталог с данными для тестирования
test_dir = 'C:\\Users\\egiazaryan\\Desktop\\dataset2\\test'
# Размеры изображения
img_width, img_height = 14, 14
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 20
# Размер мини-выборки
batch_size = 4
# Количество изображений для обучения
nb_train_samples = 20
# Количество изображений для проверки
nb_validation_samples = 8
# Количество изображений для тестирования
nb_test_samples = 8

datagen = ImageDataGenerator(rescale=1. / 255)



model = Sequential()


model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(196))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(14, 14),
    batch_size= 4,
    class_mode='categorical')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(14, 14),
    batch_size= 4,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(14, 14),
    batch_size= 4,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

model_json = model.to_json()
json_file = open('golosovalka.json', 'w')

json_file.write(model_json)
json_file.close()
model.save_weights('golosovalka.h5')
print('sohranilos')
