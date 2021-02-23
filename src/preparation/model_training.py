import xgboost as xgb
import pandas as pd
from gensim import utils
import joblib
import gensim.parsing.preprocessing as gsp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import time
import pickle

### Loading Data
text_df = pd.read_csv("data/train_data.csv")
df_x = text_df['text'].astype(str)
df_y = text_df['language']

### Preprocessing Tools
filters = [
           gsp.strip_tags,
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords,
           gsp.strip_short,
           gsp.stem_text
          ]

def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s


### Tfidf-Class for Tranformation
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


### Applying tfidf
df_x = df_x.apply(lambda x: clean_text(x))
tfidf_transformer = Text2TfIdfTransformer()
tfidf_vectors = tfidf_transformer.fit(df_x).transform(df_x)

print('Data prepared')
### Run xgb
start_time = time.time()

myXGBmodel = xgb.XGBClassifier(objective='multi:softmax', verbosity='2')
# scores = cross_val_score(myXGBmodel, tfidf_vectors, df_y.astype(str), cv=5)
myXGBmodel.fit(tfidf_vectors, df_y.astype(str))
# print('Accuracy for Tf-Idf & XGBoost Classifier : ', scores.mean())

print(f'{(time.time()-start_time)/60}min')

try:
    joblib.dump(myXGBmodel, "xgboost_model.joblib.dat")
    print("joblib.dump() worked")
except:
    print("joblib.dump() didnt work")

