# -*- coding: utf-8 -*-
"""
Created in April 2026

@author: ASSIDAH Asma
         MASSON Louis
         ZIRN Léna
"""

import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import model_from_json


# ---------------- Classes ---------------- #


nom_classe=["fukuyama muscular dystrophy","hallervorden spatz disease","moyamoya disease","pachygyria cerebellar hypoplasia","walker warburg syndrome"]


# ---------------- Chargement de notre modèle ---------------- #


@st.cache_data
def load_model():
    model_architecture = 'model.json'  # Récupération de la structure du modèle
    model_weights = 'model.h5'         # Récupération des poids du modèle
    model = 	model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)
    return model

with st.spinner('Model is being loaded..'):
  model=load_model()
 
st.write("""
         # Classification des cerveaux         """
         )
 
file = st.file_uploader("Upload the image to be classified", type=["jpg", "png", "jpeg"])
# st.set_option('deprecation.showfileUploaderEncoding', False)
 

# ---------------- Classification à partir d'une image donnée et de notre modèle ---------------- #


def upload_predict(image, model):  
        image = np.asarray(image)
        image=image.astype('float32')
        image=cv2.resize(image,(224,224))  # 224x224 pour respecter les dimenions requises pour VGG16
        image /= 255
        image = image.reshape([-1,224,224,3])
        prediction = model.predict(image)
        pred_class=np.argmax(prediction, axis=1)       
        return pred_class, prediction
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    pred_class, prediction = upload_predict(image, model)
    st.write("The image is classified as",nom_classe[pred_class[0]])
    st.write("The probability is", prediction[0][pred_class[0]])
