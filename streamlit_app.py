# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 22:08:41 2022

@author: Pawan
"""

import streamlit as st
import pickle

pickle_in = open("model.pkl", "rb")
classifier = pickle.load(pickle_in)

def welcome():
    return 'Hi There, Welcome to the Retailer Rating Predicton App'

def predict_function(recency, frequency, monetary):
    prediction = classifier.predict([[recency, frequency, monetary]])
    final_output = int(prediction)
    return "The rating of the retailer is" + " "+str(final_output)+" "+ "out of 10."
    
    
    
    
def main():
    st.title('Retailer Rating Predictor')
    html = """
    <div style="background-color:skyblue; padding:10px"
    <h2 style="color:white; text-align:center;"> Steamlit Retailer Rating Prediction ML App </h2>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    recency = st.text_input('recency')
    frequency = st.text_input('frequency')
    monetary = st.text_input('monetary')
    result = ""
    if st.button("Predict"):
        result = predict_function(recency, frequency, monetary)
    st.success("Result: {}".format(result))
    
if __name__ == "__main__":
    main()
    
    

