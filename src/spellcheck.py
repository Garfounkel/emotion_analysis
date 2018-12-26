import pickle
import os
from multiprocessing import Pool
from autocorrect import spell


def spellcheck(word):
    return word, spell(word)


def get_spellcheck_dict(unknown_words):
    '''
    Gets a dictionary of misspeled words and their correct form.
    '''
    spellcheck_dict_path = f'pickles/spellcheck_dict.pickle'
    
    if os.path.exists(spellcheck_dict_path):
        return pickle.load(open(spellcheck_dict_path, 'rb'))

    with Pool(processes=6) as pool:
        spellchecked = list(pool.map(spellcheck, unknown_words))

    spellcheck_dict = {word: fixed_word.lower() for (word, fixed_word) in spellchecked}
    
    pickle.dump(spellcheck_dict, open(spellcheck_dict_path, 'wb'))
    
    return spellcheck_dict


'''
from spellchecker import SpellChecker

checker = SpellChecker()

def spellcheck_2(word):
    return word, checker.correction(word)
'''