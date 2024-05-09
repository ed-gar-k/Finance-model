import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load your trained model
model = joblib.load('model.pkl')

# Define categorical options as used during model training
relationship_options = ['Child', 'Head of Household', 'Other non-relatives', 'Other relative', 'Parent', 'Spouse']
marital_status_options = ['Divorced/Seperated', 'Dont know', 'Married/Living together', 'Single/Never Married', 'Widowed']
education_level_options = ['No formal education', 'Other/Dont know/RTA', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training']
job_type_options = ['Dont Know/Refuse to answer', 'Farming and Fishing', 'Formally employed Government', 'Formally employed Private', 'Government Dependent', 'Informally employed', 'No Income', 'Other Income', 'Remittance Dependent', 'Self employed']
country_options = ['Kenya', 'Rwanda', 'Tanzania', 'Uganda']

# Streamlit interface
st.title('Bank Account Prediction Model')
st.write('This predicts if a person has a bank account or not ')
# User inputs
year = st.number_input('Year', min_value=2000, max_value=2022, step=1)
st.info("if the predicted value is zero the person has no account if predicted value is 1 the person has an account")
location_type = st.radio('Location Type', ['Rural', 'Urban'])
cellphone_access = st.radio('Cellphone Access', ['Yes', 'No'])
household_size = st.number_input('Household Size', min_value=1, max_value=20)
age_of_respondent = st.number_input('Age of Respondent', min_value=16, max_value=100)
gender_of_respondent = st.radio('Gender', ['Male', 'Female'])
relationship_with_head = st.selectbox('Relationship with Head', relationship_options)
marital_status = st.selectbox('Marital Status', marital_status_options)
education_level = st.selectbox('Education Level', education_level_options)
job_type = st.selectbox('Job Type', job_type_options)
country = st.selectbox('Country', country_options)

if st.button('Predict'):
    # Prepare data for model prediction
    input_data = np.zeros((1, model.n_features_in_))  # Initialize zero array for all features
    # Assign inputs directly to positions in input_data based on how the model was trained
    # This step assumes you know the index positions of each feature as they were during training
    # Example: input_data[0, 0] = year
    # You'll need to manually encode each categorical variable into the appropriate positions in the array
    
    # Example of manual setting
    input_data[0, 0] = year
    input_data[0, 1] = 1 if location_type == 'Urban' else 0  # Assuming the second feature is 'location_type_Urban'
    input_data[0, 2] = 1 if cellphone_access == 'Yes' else 0  # and so forth
    
    # More encoding required here for other features...

    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.success(f'The predicted output is: {prediction[0]}')

