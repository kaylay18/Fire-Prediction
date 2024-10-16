import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load the label encoder
@st.cache_resource
def load_label_encoder():
    with open('label_encoder.pkl', 'rb') as label_encoder_file:
        label_encoder = pickle.load(label_encoder_file)
    return label_encoder

# Load the model and encoder
model = load_model()
label_encoder = load_label_encoder()

# Title of the app
st.title("Forest Fire Prediction")

# Sidebar inputs for user preferences
st.sidebar.header("User Preferences")

temperature = st.number_input("Enter a temperature (C):", value=0)
humidity = st.number_input("Enter humidity percentage:", value=0)
wind_speed = st.number_input("Enter wind speed (km/h):", value=0)
rainfall = st.number_input("Enter rainfall amount (mm):", value=0)
fuel_moisture = st.number_input("Enter fuel moisture percentage:", value=0)
vegetation_type= st.sidebar.selectbox("Vegetation Type", ['grassland', 'forest', 'shrubland'])
slope = st.number_input("Enter slope percentage", value=0)
region = st.sidebar.selectbox("Region", ['North', 'South', 'East', 'West'])
fire_size = st.number_input("Enter fire size (hectares):", value=0)
fire_duration = st.number_input("Enter fire duration (hours):", value=0)
suppression_cost = st.number_input("Enter supression cost ($):", value=0)

# Encoding the inputs manually (same encoding as in your training data)
input_data = pd.DataFrame({
    'Temperature (°C)': [temperature],
    'Humidity (%)': [humidity],
    'Wind Speed (km/h)': [wind_speed],
    'Rainfall (mm)': [rainfall],
    'Fuel Moisture (%)': [fuel_moisture],
    'Vegetation Type': [vegetation_type],
    'Slope (%)': [slope],
    'Region': [region],
    'Fire Size (hectares)': [fire_size],
    'Fire Duration (hours)': [fire_duration],
    'Suppression Cost ($)': [suppression_cost]
})

# One-hot encode the input data (ensure it matches the training data)
input_encoded = pd.get_dummies(input_data)

# Align columns with the training data (required columns)
required_columns = model.feature_names_in_  # Get the feature columns from the model
for col in required_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[required_columns]

# Make the prediction
prediction = model.predict(input_encoded)[0]

# Reverse the label encoding (map the prediction back to the coffee type)
fire_occurance = label_encoder.inverse_transform([prediction])[0]

# Display the prediction
st.subheader(f"Fire occurance likely?: {fire_occurance}")
