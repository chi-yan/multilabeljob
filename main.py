import tensorflow_hub as hub
import streamlit as st

import numpy as np
import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance

@st.cache
def load_hub():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

@st.cache
def load_classifier():
    print('1')
    urllib.request.urlretrieve("https://www.dropbox.com/s/144rw3sur8lyo53/classifier.pkl?dl=0", "classifier.pkl") #too big to upload to Github
    print('2')
    with open('classifier.pkl', 'rb') as fp:
        classifier = pickle.load(fp)
    return embeddings

embed = load_hub()
classifier = load_classifier()


    
st.markdown("""Test""", unsafe_allow_html=True)
sentence = st.text_input('Input job description here: ')
