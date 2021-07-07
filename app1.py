# -*- coding: utf-8 -*-
import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
from PIL import Image
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
#import SessionState
#app=Flask(__name__)
#Swagger(app)


#@app.route('/')
def welcome():
    return "Welcome All"

artifacts_path = Path.joinpath(Path.cwd(),'model_artifacts')

#@app.route('/predict',methods=["Get"])
def classify_utterance(utt):
    # load the vectorizer
    loaded_vectorizer = joblib.load(Path.joinpath(artifacts_path,'vectorizer.pickle'))

    # load the model
    loaded_model = pickle.load(Path.joinpath(artifacts_path,'classification.model','rb'))

    # make a prediction
    return(loaded_model.predict(loaded_vectorizer.transform([utt])))



def main():
    st.title("EMAIL TEMPLATE")
 
    page_bg_img = '''
             <style>
                body {
                    
                       background-image: url("https://www.stkconf.org/wp-content/uploads/2018/10/Web-Page-Background-Color.jpg");
                  background-size: cover;
      
                       }      
                </style>
            '''
   
    st.markdown(page_bg_img,unsafe_allow_html=True)
    st.header("TEMPLATES SUGGESTION ML App")
    utt = st.text_input( "USER TEXT","")
  
    result=""
    if st.button("MAIL TEMPLATE"):
        result=classify_utterance(utt)
    
        st.success('{}'.format(result))
   
if __name__=='__main__':
    main()
    
    
