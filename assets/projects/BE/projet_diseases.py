# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:02:26 2026

@author: LAL MAZ
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.metrics import confusion_matrix

es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20)

train_datagen = ImageDataGenerator(
        #rotation_range=40,
        #width_shift_range=0.2,
        #weight_shift_range=0.2,
        rescale=1./255, # normalisation des valeurs : 0-255
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #fill_mode='nearest'
        )

valid_datagen = ImageDataGenerator(
        #rotation_range=40,
        #width_shift_range=0.2,
        #weight_shift_range=0.2,
        rescale=1./255, # normalisation des valeurs : 0-255
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #fill_mode='nearest'
        )

test_datagen = ImageDataGenerator(
        #rotation_range=40,
        #width_shift_range=0.2,
        #weight_shift_range=0.2,
        rescale=1./255, # normalisation des valeurs : 0-255
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #fill_mode='nearest'
        )

test_generator = test_datagen.flow_from_directory(
    directory=r"E:/S2 cle/Apprentissage auto/tp_projet_im_classe/test/",
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=1,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

train_generator = train_datagen.flow_from_directory(
    directory=r"E:/S2 cle/Apprentissage auto/tp_projet_im_classe/train/",
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_generator = valid_datagen.flow_from_directory(
    directory=r"E:/S2 cle/Apprentissage auto/tp_projet_im_classe/val/",
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

#data_diziz = pd.read_csv("C:/Users/ML/Desktop/tp_projet_im/disease_summary.csv")

# Construction du modèle

NB_CLASSES = 5;

#build the model
model = models.Sequential()
model.add(layers.Convolution2D(10, (3, 3), padding= 'same', 
           input_shape=(256,256,1), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Convolution2D(20, (3, 3), padding= 'same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Convolution2D(30, (3, 3), padding= 'same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Convolution2D(40, (3, 3), padding= 'same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Flatten()) 

model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(NB_CLASSES, activation="softmax"))

# summary of the model
model.summary()

# compiling the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# page 3/5/9
                         
EPOCHS = 40
#BATCH_SIZE = 128 
VERBOSE = 1
history=model.fit(train_generator, epochs=EPOCHS, verbose=VERBOSE, validation_data=valid_generator,callbacks=[es]
)

##evaluate the model
test_loss, test_acc = model.evaluate(test_generator)

print('Test accuracy:', test_acc)
print(history.history.keys())
plt.figure()
plt.plot(history.history['loss'], label='test')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel("Epochs")
plt.title("Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='test')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel("Epochs")
plt.title("Accuracy")
plt.legend()
plt.show()
