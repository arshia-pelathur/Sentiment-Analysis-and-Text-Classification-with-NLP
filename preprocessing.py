import re
import nltk
from nltk.corpus import stopwords 
nltk.download('wordnet')
import string
import json
import numpy as np
from nltk.stem import WordNetLemmatizer


def removal_html(record):
    return re.sub(r'<.*?>','',record)
def removal_url(record):
    clean_text = re.sub(r'https?:\/\/[^\s]+','',record)
    return re.sub(r'www\.[a-z]?\.?(com)+|[a-z]*?\.?(com)+','',clean_text)
def twitter_handles(record):
    return re.sub(r'@\w*','',record)


with open('abbreviation_dict.json', 'r', encoding='utf-8') as f:
    abbreviation_dict = json.load(f)
with open('emoticons.json', 'r') as f:
    emoticons_dict= json.load(f)

stop_words = set(stopwords.words('english'))
negations = ["not", "no", "never", "n't"]
custom_stopwords = stop_words.difference(negations)

punctuation = set(string.punctuation)

def replace_slang(tokens):
    new_token = []
    for token in tokens:
        if token in abbreviation_dict.keys():
            replacement = abbreviation_dict[token].lower().replace(',','').split(' ')
            new_token.extend(replacement)
        else:
            new_token.append(token)
    return new_token

def convert_emoticons(tokens):
    new_token = []
    for token in tokens:
        if token in emoticons_dict.keys():
            replacement = emoticons_dict[token].lower().replace(',','').split(' ')
            new_token.extend(replacement)
        else:
            new_token.append(token)
    return new_token

def clean_tokens(token):
    return [word for word in token if word not in custom_stopwords and word not in punctuation]

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess(record):
    record = twitter_handles(record)          # handling twitter handles
    record = removal_url(record)             # removal of urls and www sites
    record = removal_html(record)            # removal of html tags
    # handling punctuations
    record = re.sub(r"[^a-z\s\-:\\\/\];='#]", ' ', record)

    # removing hashtage but keep the data in it
    record = re.sub(r'#','',record)

    # removing numeric terms
    record = re.sub(r'[0-9]+','',record)

    # removing brackets
    record = re.sub(r'\[.*?\]|\(.*?\)','',record)

    # hanlding multiple spaces
    record = re.sub(r'\s+',' ',record)

    # converts string to list
    words = record.lower().split()
    words = replace_slang(words)
    words = convert_emoticons(words)
    words = clean_tokens(words)
    words = lemmatize_tokens(words)

    return words     # returns a list
