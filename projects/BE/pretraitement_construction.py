# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:02:26 2026

@author: ASSIDAH Asma
         MASSON Louis
         ZIRN Léna
"""

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    )
from tensorflow.keras import datasets, layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import VGG16


es= EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)

checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
cb = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


# ==================== Préparation des données ==================== #


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


# ==================== Data Augmentation ==================== #



# 1e essai :
    
datagen_train = ImageDataGenerator(
    rescale = 1./ 255,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    rotation_range=2,
    brightness_range=[0.9, 1.1],
    zoom_range=[0.9, 1.1]
)

# accuracy : 0.98333
"""

# 2e essai :
 
datagen_train = ImageDataGenerator(
        rescale = 1./ 255,
        width_shift_range=0.02,
        height_shift_range=0.02,
        horizontal_flip=True,
        rotation_range=2,
        brightness_range=[0.3, 1.7],
        zoom_range=[0.9, 1.1]
    )

# accuracy : 0.97


# 3e essai :
    
datagen_train = ImageDataGenerator(
    rescale = 1./ 255,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True,
    rotation_range=2,
    # brightness_range=[0.3, 1.7],
    zoom_range=[0.7, 1.3]
)

# accuracy : 0.99
"""


# ==================== Générateur de data ==================== #


test_generator = test_datagen.flow_from_directory(
    directory=r"E:/S2 cle/Apprentissage auto/tp_projet_im_classe/test/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
 
)

train_generator = datagen_train.flow_from_directory(
    directory=r"E:/S2 cle/Apprentissage auto/tp_projet_im_classe/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_generator = valid_datagen.flow_from_directory(
    directory=r"E:/S2 cle/Apprentissage auto/tp_projet_im_classe/val/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


# ==================== Construction du modèle : VGG16 : Feature Extraction ==================== #


NB_CLASSES = 5;
 
"""
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# Include_top=False: pour choisir uniquement les couches convolutives
# weights='imagenet': pour choisir les paramètres entraînés avec la base ImageNet
# il est possible d’adapter input_shape au format des images en entrée de votre réseau

base_model.trainable = False # Pour que les paramètres des couches convolutives ne soient pas modifies.
base_model.summary()

model = models.Sequential([
    base_model,
    layers.Flatten(), # ou layers.GlobalAveragePooling2D() suivant le cas
    layers.Dense(75, activation='relu'),
    layers.Dense(25, activation='relu'),
    #layers.Dropout(0.3),
    
    # softmax dense classifier
    layers.Dense(NB_CLASSES, activation="softmax")
])

# Summary of the model
model.summary()
"""


# ==================== Construction du modèle : VGG16 : Fine Tuning ==================== #


# Construction du modèle : VGG16 : Fine Tuning
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.summary()


# Les paramètres des 17 premières couches sont gelés
for layer in base_model.layers[:17]:
    layer.trainable = False

# Pour voir quelles sont les couches entraînables
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.trainable)
     
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256,activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
        
    layers.Dense(NB_CLASSES, activation='softmax') # sortie
])

model.summary()

"""
# Résultats
pour layer 17 :
-256/128/64 : 0.971
-28/64 : 0.9733
-256/128 : 0.98

pour 256/128 :
-layer : 14 -> pas bon 0.2
-layer : 15 -> erreur
-layer : 17 -> 0.98
-layer : 16 -> 0.95999
"""


# ==================== Compilation ==================== #


# Compilation du modèle
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 15
VERBOSE = 1

history=model.fit(train_generator, epochs=EPOCHS, verbose=VERBOSE, validation_data=valid_generator,callbacks=[cb])
# models.load_model(checkpoint_filepath)


# ==================== Évalutation du modèle ==================== #


test_loss, test_acc = model.evaluate(test_generator)

print('Test accuracy:', test_acc)
print(history.history.keys())
plt.figure()
plt.plot(history.history['loss'], label='test')
plt.plot(history.history['val_loss'], label='validation')
plt.plot(history.history['accuracy'], label='test')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel("Epochs")
plt.title("Loss et Accuracy")
plt.legend()
plt.show()
"""
plt.figure()
plt.plot(history.history['accuracy'], label='test')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel("Epochs")
plt.title("Accuracy")
plt.legend()
plt.show()
"""


# ==================== Sauvegarde du modèle ==================== #


"""
model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)
    model.save_weights('model.h5')
"""




