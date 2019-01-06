import pandas as pd
from sklearn.pipeline import Pipeline
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# custom imports
from .preprocess import PipelinePreprocessor
from .embeddings import sequences_to_index
from .utils import emotion2label


def load_dataset(filepath, has_labels=True):
    '''
    Load texts and targets from csv file.
    '''
    tweet = pd.read_csv(filepath, encoding='utf-8',sep='\t')
    text = tweet['turn1'] + ' <eos> ' + tweet['turn2'] + ' <eos> ' + tweet['turn3']

    if has_labels:
        labels = tweet['label'].apply(lambda x: emotion2label.get(x, 3))
        return text, labels
    else:
        return text, tweet


def load_datasets_and_vocab_pipeline():
    '''
    Load train and test datasets, preprocess them, creates a common vocabulary and computes the max length.
    '''
    train_file = 'data/train.txt'
    test_file = 'data/true_test.txt'

    X_train, y_train = load_dataset(train_file)
    X_test, y_test = load_dataset(test_file)

    pipeline = Pipeline([('preprocess', PipelinePreprocessor())])

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.fit_transform(X_test)

    max_len = 0

    vocab = set()
    for seq in tqdm(pd.concat([X_train, X_test]), desc="Building vocabulary..."):
        vocab.update(seq)
        if len(seq) > max_len:
            max_len = len(seq)

    return (X_train, y_train), (X_test, y_test), (vocab, max_len)


def load_submission_dataset(filepath):
    '''
    Load a dataset without labels from a csv file and preprocess it.
    '''
    X_test, df_test = load_dataset(filepath, has_labels=False)
    pipeline = Pipeline([('preprocess', PipelinePreprocessor())])
    X_test = pipeline.fit_transform(X_test)
    return X_test, df_test


def train_test_val_split(X, y, final=False):
    '''
    Split train set in different ratios depending on the task.
    '''
    if final:
        train_ratio = 0.95

        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=(1 - train_ratio))

        return (x_train, y_train), (x_val, y_val)
    else:
        train_ratio = 0.7

        x_train, x_rest, y_train, y_rest = train_test_split(X, y, test_size=(1 - train_ratio))

        return (x_train, y_train), (x_rest, y_rest)