import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from preprocessing import preprocess
import string
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords 
nltk.download('wordnet')



# Streamlit app
st.title("Twitter Sentiment Analysis")

# Input tweet
tweet = st.text_input("Enter a tweet for sentiment analysis")

st.write("Your tweet is:\n\n",tweet)
# Preprocessing input tweet
# CleanTweet = preprocess(tweet.lower())  # returns a list of strings

# Model files and vectorizer files mapping
model_files = {
    "Count Vectorizer Unigram (SVM)": ("models/Model_CountVec_Unigram.pkl", "vectorizers/CountVectorizer_Unigram.pkl"),
    "TF-IDF Unigram+Bigram (SVM)": ("models/Model_TFIDF_UnigramBigram.pkl", "vectorizers/TFIDFVectorizer_UnigramBigram.pkl"),
    "One-Hot Unigram+Bigram (Naive Bayes)": ("models/Model_OneHot_UnigramBigram_NB.pkl", "vectorizers/OneHot_UnigramBigram.pkl"),
    "One-Hot Unigram+Bigram (SVC)": ("models/Model_OneHot_UnigramBigram_SVC.pkl", "vectorizers/OneHot_UnigramBigram.pkl"),
    "Word2Vec CBOW (SVM)": ("models/Model_Word2Vec_CBOW.pkl", "vectorizers/Word2Vec_CBOW.model"),
    "Word2Vec Skipgram (SVM)": ("models/Model_Word2Vec_SKIPGRAM.pkl", "vectorizers/Word2Vec_Skipgram.model")
}
# Load models
models = {}
for model_name, model_path in model_files.items():
    with open(model_path[0], 'rb') as f:
        models[model_name] = pickle.load(f)

# model_choice = st.radio("Select a model for prediction", list(model_files.keys()))


# # Predict button
# if st.button("Predict Sentiment"):
#     if tweet:
#         # Preprocess tweet (with lemmatization)
#         processed_tweet = preprocess(tweet.lower())  # Assuming preprocess() returns a list of strings
#         temp = ' '.join(processed_tweet)  # Joining the list of tokens to a string for vectorization

#         # Load the corresponding vectorizer
#         model_path, vectorizer_path = model_files[model_choice]

#         if "Count Vectorizer" in model_choice or "One-Hot" in model_choice:
#             # Load the vectorizer from the .pkl file
#             with open(vectorizer_path, 'rb') as vectorizer_file:
#                 vectorizer = pickle.load(vectorizer_file)
#             tweet_vector = vectorizer.transform([temp])

#         elif "TF-IDF" in model_choice:
#             # Load the TF-IDF vectorizer from the .pkl file
#             with open(vectorizer_path, 'rb') as vectorizer_file:
#                 vectorizer = pickle.load(vectorizer_file)
#             tweet_vector = vectorizer.transform([temp])

#         elif "Word2Vec" in model_choice:
#             # Load Word2Vec model (CBOW or Skipgram)
#             word2vec_model = Word2Vec.load(vectorizer_path)
#             tweet_vector = np.mean([word2vec_model.wv[word] for word in processed_tweet if word in word2vec_model.wv], axis=0)
#             tweet_vector = tweet_vector.reshape(1, -1)  # Reshape for prediction

#         # Load the model for prediction
#         model = models[model_choice]

#         # Get the prediction from the model
#         prediction = model.predict(tweet_vector)
        
#         # Display the prediction result
#         st.markdown(f"<h2 style='font-size:24px;'>The Tweet sentiment is: <b>{prediction[0]}</b></h2>", unsafe_allow_html=True)

#     else:
#         st.write("Please enter a tweet for prediction.")

if st.button("Predict Sentiment"):
    if tweet:
        # Preprocess tweet (with lemmatization)
        processed_tweet = preprocess(tweet.lower())  # Assuming preprocess() returns a list of strings
        temp = ' '.join(processed_tweet)  # Joining the list of tokens to a string for vectorization
        
        # Dictionary to store results
        results = {"Model": [], "Prediction": []}
        
        # Loop through all models
        for model_name, (model_path, vectorizer_path) in model_files.items():
            # Load the corresponding vectorizer
            if "Count Vectorizer" in model_name or "One-Hot" in model_name:
                with open(vectorizer_path, 'rb') as vectorizer_file:
                    vectorizer = pickle.load(vectorizer_file)
                tweet_vector = vectorizer.transform([temp])

            elif "TF-IDF" in model_name:
                with open(vectorizer_path, 'rb') as vectorizer_file:
                    vectorizer = pickle.load(vectorizer_file)
                tweet_vector = vectorizer.transform([temp])

            elif "Word2Vec" in model_name:
                word2vec_model = Word2Vec.load(vectorizer_path)
                tweet_vector = np.mean([word2vec_model.wv[word] for word in processed_tweet if word in word2vec_model.wv], axis=0)
                tweet_vector = tweet_vector.reshape(1, -1)  # Reshape for prediction

            # Get the prediction from the model
            model = models[model_name]
            prediction = model.predict(tweet_vector)[0]

            # Add the appropriate emoji based on the string prediction
            if prediction == "Positive":
                emoji = "😄"  # Positive
            elif prediction == "Negative":
                emoji = "😡"  # Negative
            else:
                emoji = "🤔"  # Neutral

            # Add the model name and prediction with emoji to the results
            results["Model"].append(model_name)
            results["Prediction"].append(f"{prediction} {emoji}")

        # Convert results dictionary to a pandas DataFrame
        results_df = pd.DataFrame(results)

        # Display the results as a table
        st.table(results_df)

    else:
        st.write("Please enter a tweet for prediction.")