# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 08:48:18 2023

@author: Hosseini
"""
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache_data  # Pour mettre les données dans la cache

# Fonction pour charger le modèle vgg19 enregistré dans un fichier
def load_model():
  # model=tf.keras.models.load_model('image_classification.hdf5')
  model=tf.keras.models.load_model(r'E:\S2 cle\Apprentissage auto\tp_projet_im_classe\image_classification.hdf5')
  return model

# Afficher temporairement un message lors de l'exécution d'un bloc de code.
with st.spinner('Model is being loaded..'):
  model=load_model()
 
# Afficher un message
st.write("""
         # Image Classification
         """
         )
 
# Afficher un widget de téléchargement de fichiers
file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])

# C'est juste pour désactiver un warning
# st.set_option('deprecation.showfileUploaderEncoding', False)
 
# Fonction pour charger une image, la redimensionner, et la classer avec le modèle
def upload_predict(upload_image, model):
    
        size = (180,180)    
        image = ImageOps.fit(upload_image, size, Image.LANCZOS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        pred_class=decode_predictions(prediction,top=1)
        
        return pred_class
    
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    # pour afficher une image
    st.image(image, use_column_width=True)
    predictions = upload_predict(image, model)
    image_class = str(predictions[0][0][1])
    score=np.round(predictions[0][0][2]) 
    st.write("The image is classified as",image_class)
    st.write("The similarity score is approximately",score)
    print("The image is classified as ",image_class, "with a similarity score of",score)