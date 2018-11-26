import pandas as pd
from sklearn.pipeline import Pipeline
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# custom imports
from .preprocess import PipelinePreprocessor
from .embeddings import sequences_to_index


def load_dataset(filepath, has_labels=True):
    tweet = pd.read_csv(filepath, encoding='utf-8',sep='\t')
    text = tweet['turn1'] + " " + tweet['turn2'] + " " + tweet['turn3']

    if has_labels:
        labels = tweet['label'].apply(lambda x: {'angry': 0, 'happy': 1, 'sad': 2}.get(x, 3))
        return text, labels
    else:
        return text


def load_datasets_and_vocab_pipeline():
    train_file = 'data/train.txt'
    test_file = 'data/test.txt'

    X_train, y_train = load_dataset(train_file)
    X_test = load_dataset(test_file, has_labels=False)

    pipeline = Pipeline([('preprocess', PipelinePreprocessor(generator=False))])

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.fit_transform(X_test)

    max_len = 0

    vocab = set()
    for seq in tqdm(X_train + X_test, desc="Building vocabulary..."):
        vocab.update(seq)
        if len(seq) > max_len:
            max_len = len(seq)

    return (X_train, y_train), X_test, (vocab, max_len)
