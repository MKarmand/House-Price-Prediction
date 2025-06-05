import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load model
# -------------------------------
@st.cache_data
def load_model():
    return joblib.load("model.pkl")  # Make sure this file exists

rf_model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 House Price Prediction")
st.markdown("Enter the house specifications below:")

# User input
lb = st.number_input("Building Area (in m²)", min_value=10, max_value=2000)
lt = st.number_input("Land Area (in m²)", min_value=10, max_value=2000)
kt = st.number_input("Number of Bedrooms ", min_value=0, max_value=20)
km = st.number_input("Number of Bathrooms ", min_value=0, max_value=20)
grs = st.number_input("Number of Garages/Carports ", min_value=0, max_value=20)

# Predict button
if st.button("🎯 Predict Price"):
    # Create DataFrame from input
    new_data = pd.DataFrame([[lb, lt, kt, km, grs]], columns=['LB', 'LT', 'KT', 'KM', 'GRS'])
    
    # Prediction
    predicted_price = rf_model.predict(new_data)[0]

    # Show prediction result
    st.success(f"💰 Predicted House Price: **Rp {predicted_price:,.0f}**")
