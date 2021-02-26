# Copyright 2021, Philipp Wicke, All rights reserved.

# Install specific dependencies
import subprocess
import sys
from gensim import utils
import gensim.parsing.preprocessing as gpp
import xgboost as xgb
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
# installed xgboost==1.3.3
# installed scikit-learn==0.23.1


# Specific pre-processing transformer
class Text2TfIdfTransformer(BaseEstimator):

    def __init__(self):
        self._model = TfidfVectorizer()
        pass

    def fit(self, df_x, df_y=None):
        df_x = df_x.apply(lambda x: clean_text(x))
        self._model.fit(df_x)
        return self

    def transform(self, df_x):
        return self._model.transform(df_x)

# Preprocessing used for training the data
def clean_text(text):
    text_filters = [
        gpp.strip_tags,
        gpp.strip_punctuation,
        gpp.strip_multiple_whitespaces,
        gpp.strip_numeric,
        gpp.remove_stopwords,
        gpp.strip_short,
        gpp.stem_text
    ]
    text = text.lower()
    text = utils.to_unicode(text)
    for filter in text_filters:
        text = filter(text)
    return text


# Load language codes
lang_codes = dict()
with open("data/labels.csv", "r") as f_in:
    f_in.readline()
    lines = f_in.readlines()
    for line in lines:
        content = line.split(",")
        lang_codes[content[0]] = content[1].strip()

# Loading the estimator (xgb model)
print("Loading identifier.")
model = load("models/xgboost_model.joblib.dat")

# Loading the tranformer (tfidf transformer)
tfidf_transformer = load("models/tfidf_transformer.joblib.dat")
print("Identifier loaded.")

# Evaluate user's file
inFile = sys.argv[1]

with open(inFile, "r", encoding="utf-8") as f:
    text = f.readlines()

print("Evaluating input file.")
input_text = [clean_text(x) for x in text]
input_text = [" ".join(input_text)]
input_text = tfidf_transformer.transform(input_text)

# Predict language of each line
print("Identifying language of document.")
y_pred = model.predict(input_text)


# Output best guess
try:
    print("---> Identified language of the document: " + lang_codes[y_pred[0]])
except KeyError:
    print("Language could not be identified. Sorry.")
