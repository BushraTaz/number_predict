#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python-headless


# In[2]:


import streamlit as st
import cv2
import numpy as np
import joblib

# Load the trained RandomForestClassifier model
loaded_rf_model = joblib.load('rf_model.joblib')

# Define a function to preprocess the image
def preprocess_image(image):
    resized_image = cv2.resize(image, (28, 28))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    flattened_image = grayscale_image.flatten()
    return flattened_image

# Create a Streamlit application
st.title('Number Prediction App')

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Read as grayscale directly
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction using the loaded model
        prediction = loaded_rf_model.predict([preprocessed_image])[0]

        # Display the prediction
        st.write(f"Prediction: {prediction}")

    except Exception as e:
        st.error(f"An error occurred: {e}")


# In[ ]:




