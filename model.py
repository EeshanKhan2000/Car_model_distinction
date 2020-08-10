import zipfile

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob, os, random
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt



def preprocess_input(x):
    
    x = tf.image.random_hue(x, 0.2)
    x = tf.image.random_saturation(x, 0.8, 1.4)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    
    return x

base_path = '../content/cropped_car_train'

val_path = '../content/original/cropped_car_train'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2,
    preprocessing_function=preprocess_input
    
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.5
)

train_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    seed=0
)

validation_generator = test_datagen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=0
)

#Initialize MobileNetV2 model
import tensorflow as tf
IMG_SHAPE = (224,224,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, 
                                               weights='imagenet')

# Fine tune from this layer onwards.
fine_tune_at = 75

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  
  keras.layers.Dense(1024,activation='relu'),
  #keras.layers.Dense(512,activation='relu'),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(196, activation='softmax')
])

print(base_model.summary())
#Roughly 54 layers



adam = keras.optimizers.Adam(lr=0.0002)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])



!gdown https://drive.google.com/uc?id=1j_3Gx-MIF5LUs1tcK5tCTOepVb6EVCV0

base_dir='model_categorical_notune_50epoch.h5'
import tensorflow as tf 
model = tf.keras.models.load_model(base_dir)

batch_size = 32
epochs = 1
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator, 
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs, 
                              workers=4,
                              validation_data=validation_generator, 
                              validation_steps=validation_steps)

IMG_SHAPE = (224,224,3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, 
                                               weights='imagenet')


base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 30

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
  
  
model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()

model.save('model_categorical_2setshalves_50epochs_10epochs.h5')
