from .dataset import load_submission_dataset
from .evaluate import label2emotion, get_predictions
from .embeddings import sequences_to_index
from .models import EnsembleModel
from .utils import fix_thresholds

import numpy as np


def generate_predictions(model, submission_file_path, word_index=None, targets=None, include_probas=False, use_thresholds=True):
    '''
    Generates predictions given a model and a csv file.
    '''
    print('Loading dataset...')
    X_test, df_test = load_submission_dataset(submission_file_path)

    try:
        model.is_ensemble_model()  # Ugly way to find the object's type
    except AttributeError:
        if word_index is None:
            word_index = pickle.load(open('pickles/datastories.twitter.300d.txt.word_index.pickle', 'rb'))
        X_test = sequences_to_index(X_test, word_index)

    print('Generating Predictions...')
    y_pred, proba_preds = get_predictions(model, X_test)

    df_test['label'] = np.vectorize(lambda x: label2emotion[x])(y_pred)

    if include_probas:
        df_test['probas'] = np.apply_along_axis(lambda x: str(x), 1, proba_preds)

    if use_thresholds:
        df_test['label'] = fix_thresholds(df_test['label'].copy(), proba_preds)

    with open('submission.txt', 'w') as file:
        df_test.to_csv(path_or_buf=file, sep='\t', index=False)
    print("Done. Wrote submission.txt file at project's root")
    
    return y_pred, proba_preds