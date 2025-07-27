import streamlit as st
import numpy as np
import pandas as pd
import joblib

#Load out saved components
model = joblib.load('gradient_boosting_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Glass Type Prediction App')
st.write('The model predicts the glass type according to the various input features below. Please enter your values: ')

#The features you outline here should match the features used during the training of the model

#Input features

RI = st.sidebar.slider('Refractive index: ', 1.5, 1.8)
Na = st.sidebar.slider('Sodium: ', 10, 18)
Mg = st.sidebar.slider('Magnessium: ', 0, 4)
Al = st.sidebar.slider('Alluminium: ', 0, 4)
Si = st.sidebar.slider('Sillicon: ', 70, 80)
K = st.sidebar.slider('Potassium: ', 0.0, 0.5)
Ca = st.sidebar.slider('Calcium: ', 5.0, 10.0)
Ba = st.sidebar.slider('Barium: ', 0, 5)
Fe = st.sidebar.slider('Iron: ', 0.0, 0.5)

#Preparing input features for the model
features = np.array([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
scaled_features = scaler.transform(features)

#Prediction
if st.button('Predict Glass Type'):
    prediction_encoded = model.predict(scaled_features)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    st.success(f'Predicted Glass Type: {prediction_label}')