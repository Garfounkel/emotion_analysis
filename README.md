# Emotion analysis
Emotion analysis of Tweets using Deep Learning. The goal is to classifify each sample into the corresponding emotion class (Sad, Happy, Angry or Others).

## Publication
We wrote a short [paper](https://aclanthology.org/S19-2051/) describing the system we submited to the competition ([pdf version here](https://aclanthology.org/S19-2051.pdf)). Please use the following bibtex entry:
```
@inproceedings{rebiai-etal-2019-scia,
    title = "{SCIA} at {S}em{E}val-2019 Task 3: Sentiment Analysis in Textual Conversations Using Deep Learning",
    author = "Rebiai, Zinedine  and
      Andersen, Simon  and
      Debrenne, Antoine  and
      Lafargue, Victor",
    booktitle = "Proceedings of the 13th International Workshop on Semantic Evaluation",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/S19-2051",
    pages = "297--301",
    abstract = "In this paper we present our submission for SemEval-2019 Task 3: EmoContext. The task consisted of classifying a textual dialogue into one of four emotion classes: happy, sad, angry or others. Our approach tried to improve on multiple aspects, preprocessing with an emphasis on spell-checking and ensembling with four different models: Bi-directional contextual LSTM (BC-LSTM), categorical Bi-LSTM (CAT-LSTM), binary convolutional Bi-LSTM (BIN-LSTM) and Gated Recurrent Unit (GRU). On the leader-board, we submitted two systems that obtained a micro F1 score (F1Î¼) of 0.711 and 0.712. After the competition, we merged our two systems with ensembling, which achieved a F1Î¼ of 0.7324 on the test dataset.",
}
```

# Usage
## Requirements
Use the requirements.txt file in order to install required libraries.

## Train a model and generate a submission file.
```
python run.py
```
Preprocess datasets, generates embeddings, train both categorical and binary models, ensemble those models and finaly generates a submission file.

## Notebook
You can checkout the `playground.ipynb` notebook which showcase our final work.

## MISC
The `language_model_playground` folder is not meant to be used "as-is", indeed it needs fastai's courses library installed. Instead it's only here to showcase some experiments we did.

# Results
Our best model scored 0.7324 micro-f1 on the test set. This model is an ensembling of our best categorical model and our best binary model. You can check results on this [page](https://competitions.codalab.org/competitions/19790#results).


