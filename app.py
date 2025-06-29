
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')



stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

st.markdown( 
    """
    <style>
    .stApp {
        background-image: url("https://griddb-pro.azureedge.net/en/wp-content/uploads/2021/11/email-spam.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    @media only screen and (max-Width: 550px){
    .stApp{
        color: white !important; 
    }
    textarea {
        background-color: #383536 !important;  
        color: white !important;            
    }
    title {
        color: black !important;  
    }
    </style>
""", unsafe_allow_html=True)



input_sms = st.text_area("Enter the Email for Prediction...")

if st.button('Predict'):

    # 1. preprocess
    msg_clean = preprocess(input_sms)
    # 2. vectorize
    msg_vector = tfidf.transform([msg_clean])
    # 3. predict
    prediction = model.predict(msg_vector)
    # 4. Display
    if prediction[0] == 1:
        st.header("🚫 Spam")
    else:
        st.header("✅ Not Spam")
