import streamlit as st
import joblib 
import numpy as np
# Load saved model and vectorizer
model = joblib.load("fraud_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("💼 Fake Job Posting Detection App")
st.write("Enter job details to check if the posting is Fraudulent or Legitimate.")

# User inputs
company_profile = st.text_area("Comapy Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")

if st.button("Predict"):
    # Combine text exactly like your notebook
    text = company_profile + " " + description + " " + requirements + " " + benefits
    
    # Transform text using saved TF-IDF
    text_vector = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(text_vector)
    
    if prediction[0] == 1:
        st.error("⚠️ This job posting is likely FRAUDULENT")
    else:
        st.success("✅ This job posting appears LEGITIMATE")