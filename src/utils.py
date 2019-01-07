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
                       'Probas': txt_probas,
                       'Text': texts.apply(lambda x: ' '.join(x)),
                      })

    if emotion:
        wrongs = np.where((y_pred != y_targets) & (y_pred == emo_idx))
    else:
        wrongs = np.where(y_pred != y_targets)

    return df.loc[wrongs]


def get_most_wrongs(y_proba, targets, texts, emotion=None):
    wrong_df = get_wrongs(y_proba, targets, texts, emotion)
    wrong_df_probas = np.vectorize(lambda x: np.fromstring(x[1:-1], sep=' '), otypes=[object])(wrong_df['Probas'].values)
    wrong_df_probas = np.asarray([x for x in wrong_df_probas])
    max_proba = np.apply_along_axis(max, 1, wrong_df_probas)
    return wrong_df.iloc[max_proba.argsort()]


def fix_thresholds(y_pred, proba_preds, threshold=0.64, final=True):
    '''
    When the classifier is not sure about a prediction (proba < 0.64)
    we classify it as others since it's the most probable class (according to class repartitions).
    '''
    for i, (angry, happy, sad, others) in enumerate(proba_preds):
        m = max(angry, happy, sad, others)
        if m < threshold and m != others:
            y_pred[i] = 'others' if final else 3
    return y_pred