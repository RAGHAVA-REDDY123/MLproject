Spam Email Classifier Using LSTM and Streamlit

------Overview--------
This project is a machine learning-based Spam Email Classifier that identifies whether an email is spam or not spam using deep learning techniques. The classifier is built with an LSTM model for natural language processing and is deployed using Streamlit for an interactive user interface.

-----Features--------

1)Text Preprocessing: Emails are preprocessed to clean the text (removal of special characters, stopwords, etc.) and tokenize it into sequences for model input.

2)Deep Learning Model:
      LSTM (Long Short-Term Memory): A type of recurrent neural network (RNN) optimized for sequence data like text, ensuring better performance in spam         
      detection.
      User Interface:
      
3) Built using Streamlit, enabling easy interaction.
      Allows users to input email text and get predictions in real time.
      
How It Works ??

-----Text Preprocessing:-------

Remove punctuation, special characters, and HTML tags.
Tokenize the text into sequences.
Pad sequences to ensure uniform length.

-----Model Training:--------

The cleaned and tokenized text is fed into an LSTM model.
The model learns to classify emails as spam or not spam based on labeled training data.

-------Deployment:--------

The trained model is integrated with a Streamlit app (app.py).
Users can interact with the app via a simple web interface to classify their email text.

-----------Technologies Used---------

Python: For preprocessing and model development. 

TensorFlow/Keras: For building and training the LSTM model.

Streamlit: For creating an interactive web application.

NLP Libraries: NLTK/Spacy for text preprocessing.

NumPy and Pandas: For data manipulation and processing.

