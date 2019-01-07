#!/usr/bin/env python

import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

from src.embeddings import get_embedding_dictionnary, get_embeddings_and_word_index, sequences_to_index
from src.dataset import load_dataset, load_datasets_and_vocab_pipeline, train_test_val_split
from src.evaluate import get_metrics, get_predictions, compare_metrics, get_metrics_binary
from src.models import model_mine, model_conv1d, EnsembleModel
from src.submission import generate_predictions
from src.model_saverloader import *
from src.utils import get_wrongs, emotion2label, label2emotion, reindex_sequences

embeddings_path = './data/embeddings/datastories.twitter.300d.txt'

os.makedirs('pickles/models/', exist_ok=True)


# ### Load and preprocess Train and Test
(X_train_txt, y_train), (X_test_txt, y_test), (vocab, max_seq_len) = load_datasets_and_vocab_pipeline()


# ### Compute the classes weights because our dataset is largely unbalanced
cls_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)


# ### Create an embedding matrix for our vocabulary
max_seq_len = 80
emb_matrix, word_index, unknown_words = get_embeddings_and_word_index(embeddings_path, max_seq_len, vocab)
max_seq_len = word_index['<max_seq_len>']
# Unknown words: 1300

print(f'Unknown words from the vocabulary: {len(unknown_words)} (previously 1300 without spellcheck fixes)')
print(f'{len(unknown_words) / len(vocab) * 100:.2f}% of our vocabulary are unknown words to the embedding matrix')


# ### Transforming our sequences of words to indexes
X_train = sequences_to_index(X_train_txt, word_index)
X_test = sequences_to_index(X_test_txt, word_index)

y_train = to_categorical(y_train, 4) if len(y_train.shape) == 1 else y_train
y_test = to_categorical(y_test, 4) if len(y_test.shape) == 1 else y_test
(x_trn, y_trn), (x_val, y_val) = train_test_val_split(X_train, y_train)


print('training set: ' + str(len(x_trn)) + ' samples')
print('validation set: ' + str(len(x_val)) + ' samples')

print('x_train:', x_trn.shape)
print('y_train:', y_trn.shape)

binarize = np.vectorize(lambda x: 0 if x in [0, 1, 2] else 1)
y_trn_2 = to_categorical(binarize(y_trn.argmax(axis=1)), 2)
y_val_2 = to_categorical(binarize(y_val.argmax(axis=1)), 2)


# ## Training categorical
model = model_mine(emb_matrix, max_seq_len)

callbacks_list = [
        ModelCheckpoint(filepath='pickles/models/best.h5', save_best_only=True, verbose=1),
        EarlyStopping(patience=4, verbose=0)
    ]


history = model.fit(x_trn, y_trn, batch_size=128, validation_data=(x_val, y_val), epochs=15, class_weight=cls_weights, callbacks=callbacks_list)


# ### Evaluating our categorical model
model = load_model('pickles/models/best.h5')
best_metrics = load_best_metrics(mode='categorical')
y_pred_test, proba_preds = get_predictions(model, X_test)
model_metrics = compare_metrics(proba_preds, y_test, best_metrics, mode='categorical')
save_model_full(model, emb_matrix, word_index, model_metrics)


# ## Training Others vs all (binary)
model_bin = model_conv1d(emb_matrix, max_seq_len, class_number=2)

callbacks_list_bin = [
        ModelCheckpoint(filepath='pickles/models/best_bin.h5', save_best_only=True, verbose=1),
        EarlyStopping(patience=4, verbose=0)
    ]


cls_weight_bin = np.array([0.5, 1.5])
cls_weight_bin

history_bin = model_bin.fit(x_trn, y_trn_2, batch_size=128, validation_data=(x_val, y_val_2), epochs=20, class_weight=cls_weight_bin, callbacks=callbacks_list_bin)


# ### Evaluate our binary model
model_bin = load_model('pickles/models/best_bin.h5')
best_bin_metrics = load_best_metrics(mode='binary')
preds_2_tst = model_bin.predict(X_test, batch_size=128)
model_bin_metrics = compare_metrics(preds_2_tst, y_test, best_bin_metrics, mode='binary')
save_model_full(model_bin, emb_matrix, word_index, model_bin_metrics)


# ## Ensemble binary and categorical
model_cat, _, word_index_cat, model_metrics_cat = load_best_model(mode='categorical')
model_bin, _, word_index_bin, model_metrics_bin = load_best_model(mode='binary')
model_ens = EnsembleModel(*load_best_model(mode='ensemble'))

model_1 = model_cat
model_2 = model_bin


# #### Test combi
preds_tst_1 = model_metrics_cat['y_proba']
preds_tst_2 = model_bin_metrics['y_proba']

combi_preds_tst = np.hstack([preds_tst_1, preds_tst_2])


# #### Train combi
reindex_sequences(X_train, word_index, word_index_cat)
trn_preds_1 = model_1.predict(X_train, batch_size=128)
reindex_sequences(X_train, word_index_cat, word_index_bin)
trn_preds_2 = model_2.predict(X_train, batch_size=128)
combi_preds_trn = np.hstack([trn_preds_1, trn_preds_2])

lreg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000).fit(combi_preds_trn, y_train.argmax(axis=1))


# ### Evaluate our ensembling model
lreg_pred_tst = lreg.predict_proba(combi_preds_tst)
best_metrics_combi = load_best_metrics(mode='ensemble')
model_combi_metrics = compare_metrics(lreg_pred_tst, y_test, best_metrics_combi, mode='ensemble')
save_ensemble_model(lreg, model_combi_metrics, [model_metrics_cat, model_metrics_bin])


# ## Generating a submission file
model = EnsembleModel(*load_best_model(mode='ensemble'))
y_pred_sub, proba_preds = generate_predictions(model, 'data/test.txt', word_index=word_index)
