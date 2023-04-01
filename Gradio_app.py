# ----- Load base libraries and packages
import gradio as gr

import numpy as np
import pandas as pd
import re

import os
import pickle



# ----- Useful lists
expected_inputs = ["gender", 
                   "SeniorCitizen", 
                   "Partner",
                   "Dependents", 
                   "tenure", 
                   "PhoneService", 
                   "MultipleLines", 
                   "InternetService", 
                   "OnlineSecurity", 
                   "OnlineBackup",
                   "DeviceProtection", 
                   "TechSupport", 
                   "StreamingTV", 
                   "StreamingMovies",
                   "Contract", "PaperlessBilling", 
                   "PaymentMethod", 
                   "MonthlyCharges", 
                   "TotalCharges"]

columns_to_scale = ["tenure", 
                    "MonthlyCharges", 
                    "TotalCharges"]

categoricals = ["gender", 
                "SeniorCitizen", 
                "Partner", 
                "Dependents", 
                "PhoneService", 
                "MultipleLines", 
                "InternetService", 
                "OnlineSecurity",
                "OnlineBackup", 
                "DeviceProtection", 
                "TechSupport", 
                "StreamingTV", 
                "StreamingMovies", 
                "Contract", 
                "PaperlessBilling", 
                "PaymentMethod"]



# ----- Helper Functions
# Function to load ML toolkit
def load_ml_toolkit(file_path="app_toolkit.pkl"):
    
    with open(file_path, "rb") as file:
        loaded_toolkit = pickle.load(file)
    return loaded_toolkit


# Importing the toolkit
loaded_toolkit = load_ml_toolkit("app_toolkit.pkl")
encoder = loaded_toolkit["encoder"]
scaler = loaded_toolkit["scaler"]
model = loaded_toolkit['model']


# Function to process inputs and return prediction
def process_and_predict(*args, encoder=encoder, scaler=scaler, model=model):
    
    
    # Convert inputs into a DataFrame
    input_data = pd.DataFrame([args], columns=expected_inputs)

    # Encode the categorical columns
    encoded_categoricals = encoder.transform(input_data[categoricals])
    encoded_categoricals = pd.DataFrame(encoded_categoricals, columns=encoder.get_feature_names_out().tolist())
    df_processed = input_data.join(encoded_categoricals)
    df_processed.drop(columns=categoricals, inplace=True)

    # Scale the numeric columns
    df_processed[columns_to_scale] = scaler.transform(df_processed[columns_to_scale])

    # Restrict column name characters to alphanumerics
    df_processed.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x), inplace=True)

    # Making the prediction
    model_output = model.predict(df_processed)
    return {"Prediction:CUSTOMER WILL CHURN": float(model_output[0]), "Prediction:CUSTOMER WILL STAY": 1-float(model_output[0])}

# Define some variable limits and lists of options
max_tenure = 200
max_monthly_charges = 400 
max_total_charges = 20000 
yes_or_no = ["Yes", "No"]
internet_service_choices = ["Yes", "No", "No internet service"]


# ----- App Interface
# Inputs
gender = gr.Dropdown(label="Gender", choices=["Female", "Male"], value="Female") # Whether the customer is a male or a female
SeniorCitizen = gr.Radio(label="Senior Citizen", choices=yes_or_no, value="No") # Whether a customer is a senior citizen or not
Partner = gr.Radio(label="Partner", choices=yes_or_no, value="No") # Whether the customer has a partner or not
Dependents = gr.Radio(label="Dependents", choices=yes_or_no, value="No") # Whether the customer has dependents or not

tenure = gr.Slider(label="Tenure (months)", minimum=1, step=1, interactive=True, value=1)# maximum= max_tenure) # Number of months the customer has stayed with the company

PhoneService = gr.Radio(label="Phone Service", choices=yes_or_no, value="Yes") # Whether the customer has a phone service or not
MultipleLines = gr.Dropdown(label="Multiple Lines", choices=["Yes", "No", "No phone service"], value="No") # Whether the customer has multiple lines or not

InternetService = gr.Dropdown(label="Internet Service", choices=["DSL", "Fiber optic", "No"], value="Fiber optic") # Customer's internet service provider
OnlineSecurity = gr.Dropdown(label="Online Security", choices=internet_service_choices, value="No") # Whether the customer has online security or not
OnlineBackup = gr.Dropdown(label="Online Backup", choices=internet_service_choices, value="No") # Whether the customer has online backup or not
DeviceProtection = gr.Dropdown(label="Device Protection", choices=internet_service_choices, value="No") # Whether the customer has device protection or not
TechSupport = gr.Dropdown(label="Tech Support", choices=internet_service_choices, value="No") # Whether the customer has tech support or not
StreamingTV = gr.Dropdown(label="TV Streaming", choices=internet_service_choices, value="No") # Whether the customer has streaming TV or not
StreamingMovies = gr.Dropdown(label="Movie Streaming", choices=internet_service_choices, value="No") # Whether the customer has streaming movies or not

Contract = gr.Dropdown(label="Contract", choices=["Month-to-month", "One year", "Two year"], value="Month-to-month", interactive= True) # The contract term of the customer
PaperlessBilling = gr.Radio(label="Paperless Billing", choices=yes_or_no, value="Yes") # Whether the customer has paperless billing or not
PaymentMethod = gr.Dropdown(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], value="Electronic check") # The customer's payment method
MonthlyCharges = gr.Slider(label="Monthly Charges")# step=0.05, maximum=max_monthly_charges) # The amount charged to the customer monthly
TotalCharges = gr.Slider(label="Total Charges" )#step=0.05, maximum=max_total_charges # The total amount charged to the customer


# Output
gr.Interface(inputs=[gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges],
             outputs = gr.Label("Submit forms to view prediction..."),
            fn=process_and_predict, 
            title= "Gradio Customer Churn Prediction Web App", 
            description= """This is App is a deployment of Customer Churn prediction Machine Learning model built with random forest """
            ).launch(inbrowser= True,
                     show_error= True)