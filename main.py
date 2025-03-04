import streamlit as st
import pickle
import json
import numpy as np

# Load the trained model and columns
with open('house_price_prediction_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]  # Extracting location names (ignoring first three cols)

# Streamlit UI
st.title("House Price Prediction App")

# User inputs
location = st.selectbox("Select Location", locations)
sqft = st.number_input("Enter Square Feet", min_value=500, max_value=10000, step=100)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)

# Prediction function
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1  # Set location column

    return model.predict([x])[0]

# Button for prediction
if st.button("Predict Price"):
    prediction = predict_price(location, sqft, bath, bhk)
    st.success(f"Predicted House Price in million: ${prediction:,.2f}")
