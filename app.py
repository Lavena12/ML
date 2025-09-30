import numpy as np
import pickle
import streamlit as st
import os

# Load the model (make sure the file is in the same folder as this script)
MODEL_PATH = "train_model.sav"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

loaded_model = load_model()

# Function for prediction
def garbage_prediction(weight_input, material_code_input):
    predicted = loaded_model.predict([[weight_input, material_code_input]])
    return predicted[0]

# Main Streamlit app
def main():
    st.title('♻️ Garbage Prediction Web App')
    st.write("Welcome to the waste category predictor!")

    # Inputs
    Weight_grams = st.number_input('Enter weight in grams', min_value=0)
    Material_code = st.number_input('Enter material code', min_value=0)

    if st.button("Predict Category"):
        Prediction = garbage_prediction(Weight_grams, Material_code)
        st.success(f"Predicted Category: **{Prediction}**")

if __name__ == '__main__':
    main()
