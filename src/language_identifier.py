# Install specific dependencies
import subprocess
import sys

def install(package):
    response = input("If package "+package+" not available, it needs to be installed (y/n) or skip (s).")
    if response == "y":
      subprocess.check_call([sys.executable, "-m", "pip", "install", package])
	elif response == "s":
	  pass
    else:
      exit("Failed to meet requirements.")

install('xgboost==1.3.3')
install('scikit-learn==0.23.1')

from gensim import utils
from collections import Counter
import gensim.parsing.preprocessing as gpp
import xgboost as xgb
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
import pandas as pd

# Specific pre-processing transformer
class Text2TfIdfTransformer(BaseEstimator):

    def __init__(self):
        self._model = TfidfVectorizer()
        pass

    def fit(self, df_x, df_y=None):
        df_x = df_x.apply(lambda x : clean_text(x))
        self._model.fit(df_x)
        return self

    def transform(self, df_x):
        return self._model.transform(df_x)

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

try:
	test_df = pd.read_csv(inFile, header=None)
except:
	exit("Could not read input file. Has to be txt or csv.")

print("Evaluating input file.")
df_test = test_df(0).astype(str)
df_test = df_test.apply(lambda x : clean_text(x))
df_test = tfidf_transformer.transform(df_test)

# Predict language of each line
print("Identifying language of document.")
y_pred = model.predict(df_test)

# Most common language identified
b = Counter(y_pred)
guess = b.most_common(1)[0][0]

# Output best guess
try:
  print("---> Identified language of the document: "+ lang_codes[guess])
except KeyError:
  print("Language could not be identified. Sorry.")
