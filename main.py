import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved model and tokenizer
model = tf.keras.models.load_model('spam_email_classifier.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocessing function
def preprocess_message(message):
    message = message.lower()
    words = word_tokenize(message)
    words = [word for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit app
st.title("Spam Detection App")

# Get user input
user_input = st.text_area("Enter a message:")

# Make prediction
if st.button("Predict"):
    if user_input:
        preprocessed_message = preprocess_message(user_input)
        sequence = tokenizer.texts_to_sequences([preprocessed_message])
        padded_sequence = pad_sequences(sequence, maxlen=50)  # Use the same maxlen as during training
        prediction = model.predict(padded_sequence)
        
        if prediction[0] > 0.5:
            st.write("This message is classified as **spam**.")
        else:
            st.write("This message is classified as **not spam**.")
    else:
        st.write("Please enter a message.")