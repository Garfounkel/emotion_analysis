from keras.models import Sequential
from keras.layers import *


def model_mine(embedding_matrix, max_seq_len, class_number=4):
    vocab_size = embedding_matrix.shape[0]
    embedding_size = embedding_matrix.shape[1]

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, weights=[embedding_matrix], 
                        input_length=max_seq_len, trainable=False, name='embedding_layer'))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(class_number, activation='softmax'))

    loss_f = 'binary_crossentropy' if class_number == 2 else 'categorical_crossentropy'
    
    model.compile(loss=loss_f, optimizer='adam', metrics=['acc'])

    print(model.summary())
    
    return model
