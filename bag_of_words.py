import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize

import string
def tokenizer_better(text):    
    punc_list = string.punctuation+'0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.lower().translate(t)
    tokens = word_tokenize(text)
    return tokens

df_train=pd.read_csv('../good_datasets/discharge/train_snippets.csv')

from nltk.corpus import stopwords
stop=list(stopwords.words('english'))

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features = 5000, stop_words = stop, tokenizer = tokenizer_better)
vect.fit(df_train['TEXT'].values)

df_test=pd.read_csv('../good_datasets/discharge/test_snippets.csv')
X_train_tf = vect.transform(df_train.TEXT.values)
X_test_tf = vect.transform(df_test.TEXT.values)

y_train = df_train.Label
y_test = df_test.Label

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(C = 0.0001, penalty = 'l2', random_state = 42)
clf.fit(X_train_tf, y_train)

model = clf
y_train_preds = model.predict_proba(X_train_tf)[:,1]
y_test_preds = model.predict_proba(X_test_tf)[:,1]