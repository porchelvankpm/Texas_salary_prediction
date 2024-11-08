import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Title of the app
st.write("""
# Texas Salary Prediction App

This app predicts the **Salary** based on different features using a pre-trained model!
""")
st.write('---')

# Load the pre-trained model and preprocessing tools
model = joblib.load('best_model_rf.pkl')  
input_scaler = joblib.load('scaling_training.pkl')
scaler = joblib.load('scaling_testing.pkl')
agency_name_encoder = joblib.load('agency_name.pkl')
class_title_encoder = joblib.load('class_title.pkl')
full_name_encoder = joblib.load('full_name.pkl')
ethnicity_encoder = joblib.load('ethnicity.pkl')
gender_encoder = joblib.load('gender.pkl')
status_encoder = joblib.load('status.pkl')

# Sidebar for user input
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    # Categorical features
    agency_name = st.sidebar.text_input('Agency Name')
    full_name = st.sidebar.text_input('Full Name')
    class_title = st.sidebar.text_input('Class Title')
    ethnicity = st.sidebar.selectbox('Ethnicity', ('HISPANIC', 'WHITE', 'BLACK', 'ASIAN', 'OTHER', 'AM INDIAN'))
    gender = st.sidebar.selectbox('Gender', ('MALE', 'FEMALE'))
    status = st.sidebar.selectbox('Status', ('CRF - CLASSIFIED REGULAR FULL-TIME', 
                                             'URF - UNCLASSIFIED REGULAR FULL-TIME', 
                                             'CRP - CLASSIFIED REGULAR PART-TIME',
                                             'CTF - CLASSIFIED TEMPORARY FULL-TIME',
                                             'URP - UNCLASSIFIED REGULAR PART-TIME',
                                             'ERF - EXEMPT REGULAR FULL-TIME'))
    
    # Numerical features
    hourly_rate = st.sidebar.slider('Hourly Rate', 10.0, 100.0, 25.0)
    hours_per_week = st.sidebar.slider('Hours per Week', 10, 60, 40)
    state_number = st.sidebar.number_input('State Number', min_value=1, max_value=100, value=10)
    years_of_service = st.sidebar.number_input('Years of Service', min_value=0, max_value=50, value=5)
    
    # Date features
    Year = st.sidebar.number_input('Year', min_value=1990, max_value=2024, value=2023)
    Month = st.sidebar.number_input('Month', min_value=1, max_value=12, value=1)
    Date = st.sidebar.number_input('Date', min_value=1, max_value=31, value=1)
    
    # Collect all input features into a dictionary
    data = {
        'agency_name': agency_name,
        'full_name': full_name,
        'class_title': class_title,
        'ethnicity': ethnicity,
        'gender': gender,
        'status': status,
        'hourly_rate': hourly_rate,
        'hours_per_week': hours_per_week,
        'state_number': state_number,
        'Year': Year,
        'Month': Month,
        'Date': Date,
        'years_of_service': years_of_service
    }
    
    # Convert the data into a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
df = user_input_features()

# Ensure the input columns match the expected columns used during training
expected_columns = ['agency_name', 'full_name', 'class_title', 'ethnicity', 'gender', 'status',
                    'hourly_rate', 'hours_per_week', 'state_number', 'Year', 'Month', 'Date', 
                    'years_of_service']

df = df[expected_columns]  # Reorder or filter columns to match expected input

# Fill missing or empty values with 'unknown' before encoding
categorical_columns = ['agency_name', 'full_name', 'class_title', 'ethnicity', 'gender', 'status']

# Apply label encoding for categorical columns
df['agency_name'] = agency_name_encoder.fit_transform(df['agency_name'])
df['full_name'] = full_name_encoder.fit_transform(df['full_name'])
df['class_title'] = class_title_encoder.fit_transform(df['class_title'])
df['ethnicity'] = ethnicity_encoder.fit_transform(df['ethnicity'])
df['gender'] = gender_encoder.fit_transform(df['gender'])
df['status'] = status_encoder.fit_transform(df['status'])

# Scale numerical features
numerical_columns = ['hourly_rate', 'hours_per_week', 'state_number', 'Year', 'Month', 'Date', 'years_of_service']
df = input_scaler.transform(df)

# Final DataFrame used for prediction
st.write("Final feature data used for prediction:")
st.write(df)

# Ensure the DataFrame is in the correct shape (1D array)
# df_final_array = df.values.reshape(1, -1)

# Use the loaded model to make predictions
prediction_scaled = model.predict(df)

# Convert the scaled prediction back to original scale
prediction_original = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

# Display prediction results
st.header('Salary Prediction')
st.write(f"The predicted monthly salary is: ${prediction_original[0][0]:,.2f}")
