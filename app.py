import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommender import clean_text,get_recommendations
from nltk.corpus import stopwords

#####
df = pd.read_csv('prog_book.csv')
import nltk
nltk.download('stopwords')
stop = stopwords.words('english')
stop = set(stop)

df['clean_Book_title']=df['Book_title'].apply(clean_text)
df['clean_Description']=df['Description'].apply(clean_text)

vectorizer = TfidfVectorizer(analyzer='word', lowercase=False)
X = vectorizer.fit_transform(df['clean_Book_title'])
title_vectors = X.toarray()

tk = 0
st.title('Programming Book Recommendation System')

col1, col2 = st.beta_columns(2)
with col1:
    book = st.text_input('Enter book name that you liked : ')
    
with col2:
    feat = st.selectbox("Select Mode : ",['Book_title', 'Rating'])
    if st.button('Recommend'):
        tk = 1

if tk == 1:
    st.success('Recommending books similar to '+book)
    rec = st.empty()
    rec = st.dataframe(get_recommendations(book, 'Book_title', df, title_vectors, feat), width=700, height=76)

