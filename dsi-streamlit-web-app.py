
# Import libraries
import streamlit as st
import pandas as pd
import joblib

#l load our model pipeline
model = joblib.load("model.joblib")

# add title and instructions
st.title("Purchase Prediction Model")
st.subheader("Enter customer informtation and submit for lielihood to purchase")

# age input form
age = st.number_input(
    label = "01. Enter the customer's age",
    min_value=18,
    max_value = 120,
    value=35
    )

# gender input form
gender =st.radio(
    label = "01. Enter the customer's gender",
    options = ['M', 'F']
    )

# credit score input form
credit_score = st.number_input(
    label = "01. Enter the customer's credit score",
    min_value=0,
    max_value = 1000,
    value=500
    )

# submit input to model
if st.button("Submit For Prediction"):
    
    # store our data in a dataframe for prediction
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender], "credit_score" : [credit_score]})
    
    # aaply model pipeline to the inout data and extract probabilityt prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # output prediction
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba:.0%}")
















