# Language Identifier

## Language Identification using XGBoost
This repository holds the code for training and application of a language identification model. The model has been trained on the [WiLI-2018 - Wikipedia Language Identification database](https://zenodo.org/record/841984). The classifier is an [XGBoost model](https://xgboost.readthedocs.io/en/latest/) and achieves an **accuracy of 85.97%** on the WiLi test dataset.

## Requirements (training the model)
- Python 3.6+
- xgboost==1.3.3
- scikit-learn==0.23.1
- gensim, sklearn, pandas, joblib

## How to run the program

```
python language_identifier.py [INPUT_FILE.txt]
```

## XGBoost Model
Specifics
