import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import PorterStemmer
import os


nltk.download('punkt_tab')
nltk.download('stopwords')
ps = PorterStemmer()


def transform_text(text):
  text = text.lower()
  try:
    nltk.data.find('tokenizers/punkt')
  except LookupError:
    nltk.download('punkt')
  text = nltk.word_tokenize(text)
  y =[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:] #mutable list so cloning not copy directly
  y.clear()
  for i in text:
    if i not in stopwords.words("english") and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

with open("require.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))
st.title("Spam SMS Detection")

input_sms = st.text_area("Enter the message")
if st.button("Predict"):

    #1 preprocessing 
    transformed_sms  = transform_text(input_sms)
    #2 vectorizing
    vector_input = tfidf.transform([transformed_sms])
    #3 predict
    result = model.predict(vector_input)[0]
    #4 display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")  

