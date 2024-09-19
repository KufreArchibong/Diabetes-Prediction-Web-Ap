import numpy as np
import pickle
import streamlit as st
from PIL import Image

# Load the model with error handling
try:
    with open(r'C:\Users\kufre\Downloads\capstone\trained_model_cap_final', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except FileNotFoundError:
    print("Model file not found. Please check the file path.")
    exit()
except pickle.UnpicklingError:
    print("Error loading the model. The file might be corrupted.")
    exit()

def diabetes_prediction(input_data):
    # Convert input data to a NumPy array and reshape for prediction
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction and get probabilities
    prediction = loaded_model.predict(input_data_reshaped)
    predicted_class_prob = loaded_model.predict_proba(input_data_reshaped)
    predicted_class_prob_percent = (predicted_class_prob[0][prediction[0]] * 100).round(2)

    # Return prediction result and corresponding actions with new lines for Streamlit using markdown
    if prediction[0] == 0 and 60 <= predicted_class_prob_percent <= 100:
        st.markdown(f'**Negative!**  \n**No confirmatory test recommended!**  \n**Predicted Probability: {predicted_class_prob_percent}%**')
    elif prediction[0] == 0 and 45 <= predicted_class_prob_percent <= 59:
        st.markdown(f'**Negative!**  \n**Confirmatory test recommended!**  \n**Predicted Probability: {predicted_class_prob_percent}%**')
    elif prediction[0] == 1 and 60 <= predicted_class_prob_percent <= 100:
        st.markdown(f'**Positive!**  \n**No confirmatory test recommended!**  \n**Predicted Probability: {predicted_class_prob_percent}%**')
    elif prediction[0] == 1 and 45 <= predicted_class_prob_percent <= 59:
        st.markdown(f'**Positive!**  \n**Confirmatory test recommended!**  \n**Predicted Probability: {predicted_class_prob_percent}%**')

# User Input Section
# Create a function
def main():
    # Give a title
    st.title('Diabetes Prediction Web App')

    # Split the page into two columns
    col1, col2 = st.columns([1, 2])  # Adjust the column width ratio

    # Display an image in the first column
    with col1:
        side_image = Image.open(r'C:\Users\kufre\Downloads\capstone\diabetes_app_image.jpg')  # Replace 'side_image.jpg' with your image path
        st.image(side_image, caption='Health Awareness', use_column_width=True)

    # Collect user inputs in the second column
    with col2:
        # Getting the input variables
        age = st.text_input('Age of Person')
        smoking_history_category = st.text_input('Smoking Category: never(0), current(1), ever(2), former(3), not current(4)')
        blood_glucose_level = st.text_input('Blood Glucose Level')
        gender_category = st.text_input('Gender Category: female(0), male(1), others(2)')
        Scaled_bmi = st.text_input('BMI Level')
        scaled_hyper = st.text_input('Hypertension Status: Non-Hypertensive(0), Hypertensive(1)')
        scaled_heart_disease = st.text_input('Heart Disease Status: No(0), Yes(1)')
        scaled_HbA1c_level = st.text_input('HbA1C Level')

        # Code for prediction
        diagnosis = ''

        # Creating a button for prediction
        if st.button('Predict'):
            diagnosis = diabetes_prediction([age, smoking_history_category, blood_glucose_level, gender_category, Scaled_bmi, scaled_hyper, scaled_heart_disease, scaled_HbA1c_level])

        st.success(diagnosis)

if __name__ == '__main__':
    main()




