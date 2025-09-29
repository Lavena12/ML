# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 23:28:32 2025

@author: USER
"""

import numpy as np
import pickle
import streamlit as st

# Load the model
loaded_model = pickle.load(open("C:/Users/USER/Desktop/BTECH/train_model.sav", 'rb'))

# Function for prediction
def garbage_prediction(input_data):
    # Unpack input
    weight_input, material_code_input = input_data

    # Get prediction from model
    predicted = loaded_model.predict([[weight_input, material_code_input]])

    # Optionally map material code to readable category
    # Or just return predicted[0] if model already returns category name
    return predicted[0]  # Return the prediction result (assuming it's a string/category)
    
# Main Streamlit app
def main():
    st.title('Garbage Prediction Web App')
    st.write("Welcome to the waste category predictor!")

    # Inputs
    Weight_grams = st.number_input('Enter weight in grams', min_value=0)
    Material_code = st.number_input('Enter material code', min_value=0)

    Prediction = ''

    if st.button("Predict Category"):
        Prediction = garbage_prediction([Weight_grams, Material_code])
        st.success(f"Predicted Category: {Prediction}")
        
if __name__ == '__main__':
    main()
    
       


   
        