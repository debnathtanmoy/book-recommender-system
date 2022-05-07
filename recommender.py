import numpy as np
import pandas as pd 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('prog_book.csv')
stop = stopwords.words('english')

stop = set(stop)

def lower(text):
    """
    Convert text to lower for uniformity
    """
    return text.lower()

def remove_punctuation(text):
    """
    remove punctuatations
    """
    return text.translate(str.maketrans('','', punctuation))

def remove_stopwords(text):
    """
    remove stopwords
    """
    return " ".join([word for word in str(text).split() if word not in stop])

def remove_digits(text):
    """
    remove digits
    """
    return re.sub(r'\d+', '', text)

def clean_text(text):
    """
    one function to perform all the text cleaning
    """
    text = lower(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = remove_digits(text)
    return text

df['clean_Book_title']=df['Book_title'].apply(clean_text)
df.head()

df['clean_Description']=df['Description'].apply(clean_text)
df.head()

# intialiazing vectorizer
vectorizer = TfidfVectorizer(analyzer='word', lowercase=False)
X = vectorizer.fit_transform(df['clean_Book_title'])
title_vectors = X.toarray()
title_vectors
desc_vectorizer = TfidfVectorizer(analyzer='word', lowercase=False)
Y = desc_vectorizer.fit_transform(df['clean_Description'])
desc_vectors = Y.toarray()
desc_vectors


def get_recommendations(value_of_element, feature_locate, df, vectors_array, feature_show):
    """Returns DataFrame with particular feature of target and the same feature of five objects similar to it.

    value_of_element     - unique value of target object
    feature_locate       - name of the feature which this unique value belongs to
    df                   - DataFrame with feautures
    vectors_array        - array of vectorized text used to find similarity
    feature_show         - feature that will be shown in final DataFrame
    """
    
    index_of_element = df[df[feature_locate]==value_of_element].index.values[0]
    show_value_of_element = df.iloc[index_of_element][feature_show]
    df_without = df.drop(index_of_element).reset_index().drop(['index'], axis=1)
    vectors_array = list(vectors_array)
    target = vectors_array.pop(index_of_element).reshape(1,-1)
    vectors_array = np.array(vectors_array)
    most_similar_sklearn = cosine_similarity(target, vectors_array)[0]
    idx = (-most_similar_sklearn).argsort()
    all_values = df_without[[feature_show]]
    for index in idx:
        simular = all_values.values[idx]
     
    recommendations_df = pd.DataFrame({feature_show: show_value_of_element,
                                    "recommendation_1": simular[0][0],
                                    "recommendation_2": simular[1][0],
                                    "recommendation_3": simular[2][0],
                                    "recommendation_4": simular[3][0],
                                    "recommendation_5": simular[4][0]}, index=[0])
    

    return recommendations_df
