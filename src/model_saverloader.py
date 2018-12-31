import os
import pickle
from keras.models import load_model


def get_model_paths(acc, f1, mode='categorical'):  # TODO: Fix mode for save_model_full and load_model_full
    model_name = f'acc_{acc:.4f}-f1_{f1:.4f}'
    model_name = f'{mode}/{model_name}/'
    model_name = f'pickles/models/{model_name}'

    emb_mat_path = f'{model_name}emb_matrix.pickle'
    word_index_path = f'{model_name}word_index.pickle'
    model_path = f'{model_name}model.h5'
    model_metrics_path = f'{model_name}model_metrics.pickle'

    return model_name, emb_mat_path, word_index_path, model_path, model_metrics_path
    

def save_model_full(model, emb_matrix, word_index, model_metrics):
    acc, f1, binary = model_metrics['acc'], model_metrics['f1'], model_metrics['binary']  # Todo: Fix model_metrics['binary'] to 'mode'

    mode = 'binary' if binary else 'categorical'
    model_name, em_path, wi_path, model_path, mm_path = get_model_paths(acc, f1, mode)

    os.makedirs(model_name, exist_ok=False)

    for file_path, obj in [(em_path, emb_matrix), (wi_path, word_index), (mm_path, model_metrics)]:
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    model.save(model_path)

    print(f'Model saved at {model_name}')


def load_model_full(acc, f1, mode='categorical'):
    if mode not in ['categorical', 'binary', 'ensemble']:
        raise ValueError("mode argument must be 'categorical', 'binary' or 'ensemble'")

    model_name, em_path, wi_path, model_path, mm_path = get_model_paths(acc, f1, mode)

    emb_matrix = pickle.load(open(em_path, 'rb'))
    word_index = pickle.load(open(wi_path, 'rb'))
    model_metrics = pickle.load(open(mm_path, 'rb'))
    model = load_model(model_path)

    print(f'Loaded model from {model_name}')

    return model, emb_matrix, word_index, model_metrics

def save_ensemble_model(model, sub_model_names, model_metrics):
    acc, f1 = model_metrics['acc'], model_metrics['f1']
    
    raise NotImplementedError  # Todo: Implement


def load_best_metrics(mode='categorical'):
    '''
    mode is either categorical, binary or ensemble
    '''
    if mode not in ['categorical', 'binary', 'ensemble']:
        raise ValueError("mode argument must be 'categorical', 'binary' or 'ensemble'")

    directory = f'pickles/models/{mode}'
    
    best_metrics = {'f1': 0}
    best_model_name = None

    direct_subdirectories = next(os.walk(directory))[1]
    for model_name in direct_subdirectories:
        metrics = pickle.load(open(f'{directory}{model_name}/model_metrics.pickle', 'rb'))
        if metrics['f1'] > best_metrics['f1']:
            best_metrics = metrics
            best_model_name = model_name
    
    print(f'Best {mode} model is {best_model_name} with f1={best_metrics["f1"]}')
    return best_metrics


def load_best_model(mode='categorical'):
    if mode not in ['categorical', 'binary', 'ensemble']:
        raise ValueError("mode argument must be 'categorical', 'binary' or 'ensemble'")
    
    metrics = load_best_metrics(mode)
    if mode in ['categorical', 'binary']:
          return load_model_full(metrics['acc'], metrics['f1'], mode)
    else:
          raise NotImplementedError  # Todo: Implement