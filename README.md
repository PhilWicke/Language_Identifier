# Language Identifier

## Language Identification using XGBoost
Code for training and application of a language identification model. The model has been trained on the [WiLI-2018 - Wikipedia Language Identification database](https://zenodo.org/record/841984). The classifier is an [XGBoost model](https://xgboost.readthedocs.io/en/latest/) and achieves an **accuracy of 85.97%** on the WiLi test dataset.

## Requirements
- Python 3.7+
- xgboost==1.3.3
- scikit-learn==0.23.1
- gensim, sklearn, pandas, joblib

## How to run the program

```
python src/language_identifier.py [INPUT_FILE.txt]
```

The input file can be any multi-lined document with text. 

## Model
The model is built using XGBoost, a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework ([Chen and He, 2021](https://mran.microsoft.com/web/packages/xgboost/vignettes/xgboost.pdf)). A similar approach of using XGBoost for multi-class text classification has been implemented by (Avishek Nag)[https://github.com/avisheknag17/public_ml_models/blob/master/bbc_articles_text_classification/notebook/text_classification_xgboost_others.ipynb]. The data has been preprocessed and piped through sklearn's (TfidfVectorizer)[https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html]. The training used the XGBoost objective of 'multi:softmax' for categorial data and took 33.5min (with an AMD Ryzen7 3700X (8-core, 16 threads)). The TfidfVectorizer is needed (scikit-learn==0.23.1) to transform new data before predicting its language label with the classifier, therefore the TfidfVectorizer is provided as ```tfidf_transformer.joblib.dat``` joblib archive. The XGBoost model (xgboost==1.3.3) is provided as ```xgboost_model.joblib.dat``` joblib archive.

## Data
The [WiLI-2018 - Wikipedia Language Identification database](https://zenodo.org/record/841984) contains 235000 paragraphs of 235 languages. Language labels for identification and prediction, as well as links to train and test data are provided in the data folders. The languages range from Achinese, Afrikaans and Alemannic German to Yiddish, Yoruba to Zeeuws.

