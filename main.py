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


    
st.markdown("""Test""", unsafe_allow_html=True)
sentence = st.text_input('Input job description here: ')

if sentence:
    st.markdown(sentence)
