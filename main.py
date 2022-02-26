import tensorflow_hub as hub
import streamlit as st
import urllib.request
import pickle
import numpy as np
import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance
import os

@st.cache
def load_hub():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

@st.cache
def load_classifier():
    print('1')
    os.system('wget -O classifier.pkl.gz https://github.com/chi-yan/notebooks/blob/master/classifier.pkl.gz?raw=true')
    os.system('gunzip classifier.pkl.gz')  
    print('2')
    with open('classifier.pkl', 'rb') as fp:
        classifier = pickle.load(fp)
    return classifier

embed = load_hub()
classifier = load_classifier()

d = {0: 'Accounting', 1: 'Administration & Office Support', 2: 'Advertising, Arts & Media', 3: 'Banking & Financial Services', 4: 'CEO & General Management', 5: 'Call Centre & Customer Service', 6: 'Community Services & Development', 7: 'Construction', 8: 'Consulting & Strategy', 9: 'Design & Architecture', 10: 'Education & Training', 11: 'Engineering', 12: 'Farming, Animals & Conservation', 13: 'Government & Defence', 14: 'Healthcare & Medical', 15: 'Hospitality & Tourism', 16: 'Human Resources & Recruitment', 17: 'Information & Communication Technology', 18: 'Insurance & Superannuation', 19: 'Legal', 20: 'Manufacturing, Transport & Logistics', 21: 'Marketing & Communications', 22: 'Mining, Resources & Energy', 23: 'Real Estate & Property', 24: 'Retail & Consumer Products', 25: 'Sales', 26: 'Science & Technology', 27: 'Self Employment', 28: 'Sport & Recreation', 29: 'Trades & Services'}
categories = list(d.values())
    
st.markdown(f"""
<h1>Job Category Predictor</h1>
<br>
A famous job website has 30 categories: 
<br>{categories}<br>
Training data is from 
<a href="https://data.world/promptcloud/30000-job-postings-from-seek-australia">
https://data.world/promptcloud/30000-job-postings-from-seek-australia
</a><br><br>""", unsafe_allow_html=True)

sentence = st.text_area('Input job description here: ', value='You need to get good at algorithms, React and Javascript')

if st.button('Convert description to job category'):
    x_test = embed([sentence])
    y_new_prediction = classifier.predict(x_test)
    st.markdown(d[y_new_prediction.todense()[0].tolist()[0][0]])
