# Emotion analysis
Emotion analysis of Tweets using Deep Learning. The goal is to classifify each sample into the corresponding emotion class (Sad, Happy, Angry or Others).

# Requirements
Use the requirements.txt file in order to install required libraries.


# Run (dev)
```
python run.py
```
Preprocess datasets, generates embeddings, train both categorical and binary models, ensemble those models and finaly generates a submission file.

# Results (dev set)
Our best model scored 0.7099 micro-f1 on the dev set (also named test set). This model is an ensembling of our best categorical model and our best binary model. You can check results on this [page](https://competitions.codalab.org/competitions/19790#results).