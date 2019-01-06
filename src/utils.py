import numpy as np
import pandas as pd


label2emotion = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'others'}
emotion2label = {"others": 3, "happy": 1, "sad": 2, "angry": 0}

def reindex_sequences(seq_samples, src_word_index, dst_word_index):
    '''
    Reindex sequences given the source word_index and the target word_index.
    '''
    src_index_word = {v: k for k, v in src_word_index.items()}
    for seq in seq_samples:
        for i in range(len(seq)):
            idx = seq[i]
            word = src_index_word[idx]
            seq[i] = dst_word_index[word]


def check_mode(mode):
    '''
    Check mode argument for model_saver_loader.py
    '''
    if mode not in ['categorical', 'binary', 'ensemble']:
        raise ValueError("mode argument must be 'categorical', 'binary' or 'ensemble'")


def get_wrongs(y_proba, targets, texts, emotion=None):
    '''
    Get wrongly classified texts given an emotion.
    '''
    if emotion:
        emo_idx = emotion2label[emotion]

    y_pred = y_proba.argmax(axis=1)
    y_targets = targets.argmax(axis=1)

    emotions_targets = np.vectorize(lambda x: label2emotion[x])(y_targets)
    emotions_predicted = np.vectorize(lambda x: label2emotion[x])(y_pred)

    txt_probas = [str(x) for x in y_proba]
    df = pd.DataFrame({'Predicted': emotions_predicted,
                       'Actual': emotions_targets,
#                        'Probas': txt_probas,
                       'Text': texts.apply(lambda x: ' '.join(x)),
                      })

    if emotion:
        wrongs = np.where((y_pred != y_targets) & (y_pred == emo_idx))
    else:
        wrongs = np.where(y_pred != y_targets)

    return df.loc[wrongs]