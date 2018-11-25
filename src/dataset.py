import pandas as pd



def load_dataset(filepath, has_labels=True):
    tweet = pd.read_csv(filepath, encoding='utf-8',sep='\t')
    text = tweet['turn1'] + " " + tweet['turn2'] + " " + tweet['turn3']

    if has_labels:
        labels = tweet['label'].apply(lambda x: {'angry': 0, 'happy': 1, 'sad': 2}.get(x, 3))
        return text, labels
    else:
        return text