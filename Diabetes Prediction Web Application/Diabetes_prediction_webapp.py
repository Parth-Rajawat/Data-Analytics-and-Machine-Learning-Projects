import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('/home/parth/[PROGRAMMING] Data Science/Data Analytics/Complete Projects/Diabetes Prediction Web Application/trained_model.sav', 'rb'))

#creating a function for prediction
def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_nparray = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_nparray.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)

    if(prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
    
def main():
    
    #Title for the Webpage
    st.title("Diabates Prediction application")
    
    #getting the input data from the user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Levels")
    BloodPressure = st.text_input("Blood Pressure Levels")
    SkinThickness = st.text_input("Skin Thickness level")
    Insulin = st.text_input("Insulin Levels")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree function value")
    Age = st.text_input("Age of the Person")

    #Code for Prediction
    diagnosis = ""

    #creating a button for the prediction
    if st.button("Results of the test"):
       diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis) #prints the results


if __name__ == '__main__':
    main()


