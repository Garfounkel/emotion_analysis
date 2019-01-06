import os
import pickle
from keras.models import load_model

from .utils import check_mode


def get_model_paths(acc, f1, mode='categorical'):
    '''
    Get paths names given a model accuracy, f1 and mode.
    '''
    model_name = f'acc_{acc:.4f}-f1_{f1:.4f}'
    model_name = f'{mode}/{model_name}/'
    model_name = f'pickles/models/{model_name}'

    emb_mat_path = f'{model_name}emb_matrix.pickle'
    word_index_path = f'{model_name}word_index.pickle'
    model_path = f'{model_name}model.h5'
    model_metrics_path = f'{model_name}model_metrics.sav'

    return model_name, emb_mat_path, word_index_path, model_path, model_metrics_path
    

def save_model_full(model, emb_matrix, word_index, model_metrics):
    '''
    Saves a binary or categorical model.
    '''
    acc, f1, mode = model_metrics['acc'], model_metrics['f1'], model_metrics['mode']

    model_name, em_path, wi_path, model_path, mm_path = get_model_paths(acc, f1, mode)
    model_metrics['name'] = model_name

    os.makedirs(model_name, exist_ok=False)

    for file_path, obj in [(em_path, emb_matrix), (wi_path, word_index), (mm_path, model_metrics)]:
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    model.save(model_path)

    print(f'Model saved at {model_name}')


def load_model_full(acc, f1, mode='categorical'):
    '''
    Loads a binary or categorical model.
    '''
    check_mode(mode)

    model_name, em_path, wi_path, model_path, mm_path = get_model_paths(acc, f1, mode)

    emb_matrix = pickle.load(open(em_path, 'rb'))
    word_index = pickle.load(open(wi_path, 'rb'))
    model_metrics = pickle.load(open(mm_path, 'rb'))
    model = load_model(model_path)

    print(f'Loaded model from {model_name}')

    return model, emb_matrix, word_index, model_metrics

def save_ensemble_model(model, model_metrics, sub_models_metrics):
    '''
    Saves an ensemble model.
    model: ensemble model
    model_metrics: current model metrics
    sub_models_metrics: list of model_metrics
    '''
    acc, f1 = model_metrics['acc'], model_metrics['f1']
    model_name, _, _, model_path, mm_path = get_model_paths(acc, f1, 'ensemble')
    sub_mm_path = f'{model_name}sub_model_metrics.sav'

    os.makedirs(model_name, exist_ok=False)

    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(model_metrics, open(mm_path, 'wb'))
    pickle.dump(sub_models_metrics, open(sub_mm_path, 'wb'))

    print(f'Model saved at {model_name}')


def load_ensemble_model(acc, f1):
    '''
    Loads an ensemble model.
    '''
    model_name, _, _, model_path, mm_path = get_model_paths(acc, f1, 'ensemble')
    sub_mm_path = f'{model_name}sub_model_metrics.sav'

    model = pickle.load(open(model_path, 'rb'))
    model_metrics = pickle.load(open(mm_path, 'rb'))
    sub_model_metrics = pickle.load(open(sub_mm_path, 'rb'))

    print(f'Loaded model from {model_name}')
    return model, model_metrics, sub_model_metrics


def load_best_metrics(mode='categorical'):
    '''
    Loads the best metrics for a given mode.
    mode is either categorical, binary or ensemble
    '''
    check_mode(mode)

    directory = f'pickles/models/{mode}/'
    
    best_metrics = {'f1': 0}
    best_model_name = None

    direct_subdirectories = next(os.walk(directory))[1]
    for model_name in direct_subdirectories:
        metrics = pickle.load(open(f'{directory}{model_name}/model_metrics.sav', 'rb'))
        if metrics['f1'] > best_metrics['f1']:
            best_metrics = metrics
            best_model_name = model_name
    
    print(f'Best {mode} model is {best_model_name} with f1={best_metrics["f1"]}')
    return best_metrics


def load_best_model(mode='categorical'):
    '''
    Loads the best model for a given mode.
    '''
    check_mode(mode)
    
    metrics = load_best_metrics(mode)
    if mode in ['categorical', 'binary']:
          return load_model_full(metrics['acc'], metrics['f1'], mode)
    else:
          return load_ensemble_model(metrics['acc'], metrics['f1'])