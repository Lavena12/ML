import numpy as np
import joblib
import streamlit as st

# Load the model (make sure train_model.joblib is in the repo folder)
@st.cache_resource
def load_model():
    return joblib.load("train_model.joblib")

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
