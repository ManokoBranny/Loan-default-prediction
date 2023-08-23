import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from streamlit_option_menu import option_menu
import streamlit as st
import pickle

# Load the pre-trained models
import pickle
import streamlit as st
import pickle
import streamlit as st

#function to load the model
def load_model(model_name):
    return pickle.load(open(model_name, 'rb'))

# Function to make predictions
def predict(model, data):
    return model.predict(data)
# # Function to load the model from file
# def load_model(model_name):
#     return pickle.load(open(model_name, 'rb'))
import streamlit as st
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score

# Function to load the model from file
def load_model(model_name):
    return pickle.load(open(model_name, 'rb'))

# Function to make predictions
def predict(model, data):
    return model.predict(data)

# Load the models
models = {
    'LinearSVC': load_model('LinearSVC'),
    'RandomForestClassifier': load_model('RandomForestClassifier'),
    'DecisionTreeClassifier': load_model('DecisionTreeClassifier'),
    'SVC': load_model('SVC'),
    'LogisticRegression': load_model('LogisticRegression'),
    'KNeighborsClassifier': load_model('KNeighborsClassifier'),
    'GradientBoostingClassifier': load_model('GradientBoostingClassifier')
}


# Set page configuration and title
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# Add a brief description
# st.write("This app predicts whether a client is likely to default on their loan based on certain features.")


# Sidebar
with st.sidebar:
    from PIL import Image
    #image2 = Image.open('/Users/da_m1_23/Downloads/Manoko-loan-default-prediction/fraudimage.jpeg')
    image2=Image.open("fraudimage.jpeg")
    st.image(image2, caption='Loan Default Prediction')

    #page_selection = st.selectbox("Navigation", 
                                #   ["Overview", "Step 1: Model", "Step 3: Output", "Contact Us"])
    
    page_selection = option_menu(
            menu_title=None,
            options=["Overview", "Model", "Contact Us"],
            icons=['file-earmark-text', 'graph-up', 'robot', 'file-earmark-spreadsheet', 'envelope'],
            menu_icon='cast',
            default_index=0,
            # orientation='horizontal',
            styles={"container": {'padding': '0!important', 'background_color': 'red'},
                    'icon': {'color': 'red', 'font-size': '18px'},
                    'nav-link': {
                        'font-size': '15px',
                        'text-align': 'left',
                        'margin': '0px',
                        '--hover-color': '#4BAAFF',
                    },
                    'nav-link-selected': {'background-color': '#6187D2'},
                    }
        )

# Main content
if page_selection == "Overview":
    st.title('Loan Default Prediction Application')
    st.header("predicting clients that are most likely to default on their loans")
    #st.image('/Users/da_m1_23/Downloads/Manoko-loan-default-prediction/loanimage.jpeg')
    st.image('loanimage.jpeg')
    st.markdown("This app is designed to predict whether a client is likely to default on their loan based on various input features. The app uses machine learning models to make predictions and provides insights into the risk assessment process.")
    st.write("### Why use machine learning for loan default prediction?")
    st.markdown("- ** Default Prediction machine learning models are more effective than humans**")
    st.markdown("- **ML handles overload well**")
    st.markdown("- **ML beats traditional default prediction systems**")

elif page_selection == "Model":
     #Load the models
    models = {
    'LinearSVC': load_model('LinearSVC'),
    'RandomForestClassifier': load_model('RandomForestClassifier'),
    'DecisionTreeClassifier': load_model('DecisionTreeClassifier'),
    'SVC': load_model('SVC'),
    'LogisticRegression': load_model('LogisticRegression'),
    'KNeighborsClassifier': load_model('KNeighborsClassifier'),
    'GradientBoostingClassifier': load_model('GradientBoostingClassifier')
}
     #Dropdown to select the model
    selected_model = st.selectbox("Select Model", list(models.keys()))

    st.title("Loan Default Prediction - Step 1: Input")
    st.write("Enter the client's information:")
    loannumber = st.number_input("Loan Number", value=1, min_value=1)
    loanamount = st.number_input("Loan Amount", value=1000.0, min_value=0.0, step=100.0)
    totaldue = st.number_input("Total Due", value=1200.0, min_value=0.0, step=100.0)
    termdays = st.number_input("Term Days", value=30, min_value=1)
    days_in_advance = st.number_input("Days in Advance", value=0, min_value=-30, step=1)
    due_day = st.number_input("Due Day of Month", value=1, min_value=1, max_value=31)
    birth_year = st.number_input("Birth Year", value=1990, min_value=1900, max_value=2023)
    age = st.number_input("Age", value=30, min_value=1)
    # Make prediction using the selected model
    if st.button("Predict"):
        model = models[selected_model]
        input_data = [[loannumber, loanamount, totaldue, termdays, days_in_advance, due_day, birth_year, age]]
        prediction = predict(model, input_data)
        result = "likely to default" if prediction[0] == 1 else "not likely to default"
        st.write(f"Prediction: {prediction[0]}")
        st.write(f"Prediction: The client is {result}.")
elif page_selection == "Contact Us":
        st.title('Contact Us!')
        st.markdown("Have a question or want to get in touch with us? Please fill out the form below with your email "
                    "address, and we'll get back to you as soon as possible. We value your privacy and assure you "
                    "that your information will be kept confidential.")
        st.markdown("By submitting this form, you consent to receiving email communications from us regarding your "
                    "inquiry. We may use the email address you provide to respond to your message and provide any "
                    "necessary assistance or information.")
        with st.form("Email Form"):
            subject = st.text_input(label='Subject', placeholder='Please enter subject of your email')
            fullname = st.text_input(label='Full Name', placeholder='Please enter your full name')
            email = st.text_input(label='Email Address', placeholder='Please enter your email address')
            text = st.text_area(label='Email Text', placeholder='Please enter your text here')
            uploaded_file = st.file_uploader("Attachment")
            submit_res = st.form_submit_button("Send")
        st.markdown("Thank you for reaching out to us. We appreciate your interest in our loan default web "
                    "application and look forward to connecting with you soon")
       

    

# # Streamlit app code
# st.title("Loan Default Prediction")

# # Dropdown to select the model
# selected_model = st.selectbox("Select Model", list(models.keys()))

# # Input fields for the features
# loannumber = st.number_input("Loan Number", value=1, min_value=1)
# loanamount = st.number_input("Loan Amount", value=1000.0, min_value=0.0, step=100.0)
# totaldue = st.number_input("Total Due", value=1200.0, min_value=0.0, step=100.0)
# termdays = st.number_input("Term Days", value=30, min_value=1)
# days_in_advance = st.number_input("Days in Advance", value=0, min_value=-30, step=1)
# due_day = st.number_input("Due Day of Month", value=1, min_value=1, max_value=31)
# birth_year = st.number_input("Birth Year", value=1990, min_value=1900, max_value=2023)
# age = st.number_input("Age", value=30, min_value=1)

# Make prediction using the selected model
# Make prediction using the selected model
# if st.button("Predict"):
#     model = models[selected_model]
#     input_data = [[loannumber, loanamount, totaldue, termdays, days_in_advance, due_day, birth_year, age]]
#     prediction = predict(model, input_data)
#     result = "likely to default" if prediction[0] == 1 else "not likely to default"
#     st.write(f"Prediction: The client is {result}.")
#     st.write(f"Prediction: {prediction[0]}")
