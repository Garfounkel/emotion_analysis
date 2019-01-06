from keras.models import Sequential
from keras.layers import *
import pickle

from .model_saverloader import load_model_full
from .embeddings import sequences_to_index


def model_mine(embedding_matrix, max_seq_len, class_number=4):
    vocab_size = embedding_matrix.shape[0]
    embedding_size = embedding_matrix.shape[1]

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, weights=[embedding_matrix], 
                        input_length=max_seq_len, trainable=False, name='embedding_layer'))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(150, return_sequences=True, recurrent_dropout=0.5)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(150, return_sequences=True, recurrent_dropout=0.5)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(class_number, activation='softmax'))

    loss_f = 'binary_crossentropy' if class_number == 2 else 'categorical_crossentropy'
    
    model.compile(loss=loss_f, optimizer='adam', metrics=['acc'])

    print(model.summary())
    
    return model


def model_conv1d(embedding_matrix, max_seq_len, class_number=4):
    vocab_size = embedding_matrix.shape[0]
    embedding_size = embedding_matrix.shape[1]

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, weights=[embedding_matrix],
                        input_length=max_seq_len, trainable=False, name='embedding_layer'))
    model.add(Dropout(0.3))
    model.add(Conv1D(64, 5, padding='same'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(150, return_sequences=True, recurrent_dropout=0.5)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(150, return_sequences=True, recurrent_dropout=0.5)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(class_number, activation='softmax'))

    loss_f = 'binary_crossentropy' if class_number == 2 else 'categorical_crossentropy'

    model.compile(loss=loss_f, optimizer='adam', metrics=['acc'])

    print(model.summary())

    return model


class EnsembleModel():
    '''
    Serves as an api to aggregates multiple models.
    '''
    def __init__(self, model, model_metrics, sub_models_metrics):
        self.model = model
        self.model_metrics = model_metrics
        self.sub_models_metrics = sub_models_metrics
    
    def predict(self, text_sequences, batch_size=128):
        list_preds = []
        
        for metrics in self.sub_models_metrics:
            sub_model, _, sub_word_index, _ = load_model_full(metrics['acc'], metrics['f1'], metrics['mode'])
            seq_samples = pickle.loads(pickle.dumps(text_sequences))  # deep copy
            seq_samples = sequences_to_index(seq_samples, sub_word_index)
            list_preds.append(sub_model.predict(seq_samples, batch_size=batch_size))

        combi_preds = np.hstack(list_preds)
        
        predictions = self.model.predict_proba(combi_preds)

        return predictions
    
    def is_ensemble_model(self):
        return True