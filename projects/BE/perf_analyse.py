# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:38:06 2026

@author: ASSIDAH Asma
         MASSON Louis
         ZIRN Léna
"""

import tensorflow as tf
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
from tensorflow.keras.models import model_from_json
import cv2
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, classification_report 


# ================ Chargement de notre model : architecture et paramètres ================ #


model_architecture = 'model.json' # Architecture
model_weights = 'model.h5'        # Poids
model = model_from_json(open(model_architecture).read()) # Permet de récupérer el model
model.load_weights(model_weights) # Chargement des poid/paramètres qui engendrent los puntos
model.summary()


# ================ Visualisation des filtres ================ #


base_model = model.layers[0] # Récupéracion de la première couche
base_model.summary()

# Récupération des filtres pour la n.ieme couche
filters, biases, *is_anything_else_being_returned = base_model.layers[16].get_weights()

# Normalisation des filtres sur [0 , 1]
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Prendre le deuxième filtre de cette couche (qui contient 3 sous-filtres pour les 3 canaux R,
# V et et B) et visualiser ces 3 sous-filtres
f=filters[:,:,:,1]
plt.subplot(1,3,1)
plt.imshow(f[:, :, 0], cmap='gray')
plt.subplot(1,3,2)
plt.imshow(f[:, :, 1], cmap='gray')
plt.subplot(1,3,3)
plt.imshow(f[:, :, 2], cmap='gray')


# ================ Visualisation du contenu des caractéristiques extraites de la n.ieme couche ================ #


img_tst = Image.open(r"E:/S2 cle/Apprentissage auto/tp_projet_im_classe/test/hallervorden_spatz_disease\hallervorden_spatz_disease_test_0004.jpg")
image = np.asarray(img_tst)
image=image.astype('float32')
image=cv2.resize(image,(224,224))
image /= 255
image = image.reshape([-1,224,224,3])

from keras.models import Model
inter_model = Model(inputs=base_model.inputs, outputs=base_model.layers[16].output)
inter_model.summary()
feature_maps = inter_model.predict(image)
plt.figure()
plt.imshow(feature_maps[0,:,:,0])


# ================ Évaluation des performances ================ #


test_datagen = ImageDataGenerator(
        #rotation_range=40,
        #width_shift_range=0.2,
        #weight_shift_range=0.2,
        rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #fill_mode='nearest'
        )

test_generator = test_datagen.flow_from_directory(
    directory=r"E:/S2 cle/Apprentissage auto/tp_projet_im_classe/test/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,       # On donne à l'apprentissage image par image
    class_mode="categorical",
    shuffle=False,      # On ne tire pas aléatoirement les images
)


y_pred_prob=model.predict(test_generator)    # Création du vecteur de proba.
y_pred=np.argmax(y_pred_prob, 1)             # Sélection de la val. max
labels = test_generator.classes              # Récupération des classes
print(confusion_matrix(labels, y_pred))      # Matrice de confusion
print(classification_report(labels, y_pred)) # Affiche les performances
