import os
import numpy as np
import pickle
import itertools


'''
# TODO:
-------
[X] Convert datastories.txt to embedding dictionnary and dump the resulting embedding
[ ] Enrich the embedding dictionnary with external ressources (Emolex, OLE, Emoji Valence, Depeche Mood, ...)
[X] Build the embedding matrix corresponding to our vocabulary using the embedding dictionnary
[X] Add the vocabulary to the call of get_embedding_and_word_index(...)
'''

def get_embedding_dictionnary(filepath, dim=300):
    emb_dict_path = f'pickles/{os.path.basename(filepath)}.dict.pickle'

    if os.path.exists(emb_dict_path):
        return pickle.load(open(emb_dict_path, 'rb'))

    emb_dict = dict()
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            word_vec = line.split()
            word = word_vec[0]
            vec = np.asarray(word_vec[1:], dtype='float32')

            emb_dict[word] = vec

    emb_dict['<unk>'] = np.random.uniform(low=-0.05, high=0.05, size=dim)
    emb_dict['<pad>'] = np.zeros(dim)

    pickle.dump(emb_dict, open(emb_dict_path, 'wb'))
    return emb_dict


def enrich_embedding_dictionnary(emb_dict):
    return emb_dict


def get_embeddings_and_word_index(filepath, vocab=None, dim=300):
    emb_mat_path = f'pickles/{os.path.basename(filepath)}.matrix.pickle'
    word_index_path = f'pickles/{os.path.basename(filepath)}.word_index.pickle'

    if os.path.exists(emb_mat_path) and os.path.exists(word_index_path):
        return pickle.load(open(emb_mat_path, 'rb')), pickle.load(open(word_index_path, 'rb'))

    emb_dict = get_embedding_dictionnary(filepath, dim)
    emb_dict = enrich_embedding_dictionnary(emb_dict)

    word_number = len(emb_dict)

    word_index = dict()
    emb_matrix = np.ndarray((word_number, dim), dtype='float32')
    
    for i, (word, vec) in enumerate(emb_dict.items()):
        word_index[word] = i
        emb_matrix[i] = vec

    i = word_number

    if vocab:
        for word in vocab:
            if word not in word_index:
                word_index[word] = i
                emb_matrix = np.vstack([emb_matrix, emb_dict['<unk>']])
                i += 1
        print(f'Unknown words from the vocabulary: {i - word_number}')

    pickle.dump(emb_matrix, open(emb_mat_path, 'wb'))
    pickle.dump(emb_dict, open(word_index_path, 'wb'))

    return emb_matrix, word_index


def sequences_to_index(sequences, word_index, max_len):
    for i, seq in enumerate(sequences):
        for j, word in enumerate(seq):
            index = word_index.get(word, word_index['<unk>'])
            sequences[i][j] = index
        pad_len = max_len - len(seq)
        sequences[i] += list(itertools.repeat(word_index['<pad>'], pad_len))
    return np.array(sequences.tolist())
