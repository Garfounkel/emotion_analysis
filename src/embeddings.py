import os
import numpy as np
import pickle
import itertools
from collections import defaultdict


'''
# TODO:
-------
[X] Convert datastories.txt to embedding dictionnary and dump the resulting embedding
[ ] Enrich the embedding dictionnary with external ressources (Emolex, OLE, Emoji Valence, Depeche Mood, ...)
[X] Build the embedding matrix corresponding to our vocabulary using the embedding dictionnary
[X] Add the vocabulary to the call of get_embedding_and_word_index(...)
'''

def get_corresponding_closing_tag(emb_dict, tag):
    '''
    This is used to compute a slightly better vector for missing closing tags.
    Example: A closing tag for "<hashtag>" is "</hashtag>", 
    therefore the difference between "<hashtag>" and "</hashtag>" should be close 
    to the difference of any tag "<tag>" and it's closing tag "</tag>".
    
    This relies on the property of word2vec to preserve semantic and syntactic
    relationships through arithmetic (ex: brother - man + woman = sister).
    '''
    close_tag_difference = emb_dict['<hashtag>'] - emb_dict['</hashtag>']
    return emb_dict[tag] - close_tag_difference


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
    emb_dict['<eos>'] = np.random.uniform(low=-0.05, high=0.05, size=dim)
    emb_dict['</allcaps>'] = get_corresponding_closing_tag(emb_dict, '<allcaps>')
    emb_dict['<pad>'] = np.zeros(dim)

    pickle.dump(emb_dict, open(emb_dict_path, 'wb'))
    return emb_dict


def enrich_embedding_dictionnary(emb_dict):
    return emb_dict


def get_embeddings_and_word_index(filepath, max_seq_len, vocab, dim=300):
    emb_mat_path = f'pickles/{os.path.basename(filepath)}.matrix.pickle'
    word_index_path = f'pickles/{os.path.basename(filepath)}.word_index.pickle'
    unknown_words_path = 'pickles/unknown_words.pickle'

    if os.path.exists(emb_mat_path) and os.path.exists(word_index_path) and os.path.exists(unknown_words_path):
        return pickle.load(open(emb_mat_path, 'rb')), pickle.load(open(word_index_path, 'rb')), pickle.load(open(unknown_words_path, 'rb'))

    emb_dict = get_embedding_dictionnary(filepath, dim)
    emb_dict = enrich_embedding_dictionnary(emb_dict)

    vocab.update(['<unk>', '<pad>', '<eos>', '</allcaps>'])
    word_number = len(vocab)

    word_index = dict()
    emb_matrix = np.ndarray((word_number, dim), dtype='float32')
    
    i = 0
    unknown_words = defaultdict(int)
    if vocab:
        for word in vocab:
            word_index[word] = i
            emb_matrix[i] = emb_dict.get(word, emb_dict['<unk>'])
            i += 1
            if word not in emb_dict:
                unknown_words[word] += 1
        print(f'Unknown words from the vocabulary: {len(unknown_words)}')

    word_index['<max_seq_len>'] = max_seq_len

    pickle.dump(emb_matrix, open(emb_mat_path, 'wb'))
    pickle.dump(word_index, open(word_index_path, 'wb'))
    pickle.dump(unknown_words, open(unknown_words_path, 'wb'))

    return emb_matrix, word_index, unknown_words


def sequences_to_index(sequences, word_index, max_len):
    for i, seq in enumerate(sequences):
        for j, word in enumerate(seq):
            index = word_index.get(word, word_index['<unk>'])
            sequences[i][j] = index
        pad_len = max_len - len(seq)
        sequences[i] += list(itertools.repeat(word_index['<pad>'], pad_len))
    return np.array(sequences.tolist())
