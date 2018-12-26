from keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

label2emotion = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'others'}
emotion2label = {"others": 3, "happy": 1, "sad": 2, "angry": 0}

def plot_confusion_matrix(confusion_matrix, class_names, figsize = (7,5), fontsize=14, ax=None, title='Confusion matrix'):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    if not ax:
        fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap=plt.cm.Blues, annot_kws={"size": fontsize}, ax=ax)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    heatmap.set_ylabel('True label', fontsize=fontsize)
    heatmap.set_xlabel('Predicted label', fontsize=fontsize)
    heatmap.set_title(title, fontsize=fontsize)


def plot_2_cm(cm1, cm2, class_names, titles):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,5))
    plot_confusion_matrix(cm1, class_names, ax=ax1, title=titles[0])
    plot_confusion_matrix(cm2, class_names, ax=ax2, title=titles[1])
    fig.tight_layout()


def get_metrics(predictions, ground, NUM_CLASSES=4, print_all=True):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    print()
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(0, NUM_CLASSES - 1):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    print()

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[:-1].sum()
    falsePositives = falsePositives[:-1].sum()
    falseNegatives = falseNegatives[:-1].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    print()
    if print_all:
        print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    
    cm = confusion_matrix(ground, predictions)

    return accuracy, microPrecision, microRecall, microF1, cm


def get_predictions(model, id_sequences):
    predictions = model.predict(id_sequences, batch_size=128)
    y_pred = np.argmax(predictions, axis=1)

    return y_pred


def compare_metrics(model, id_sequences, targets, compare_dict):
    y_pred = get_predictions(model, id_sequences)

    accuracy, microPrecision, microRecall, microF1, cm = get_metrics(predictions, targets, print_all=False)

    print("Accuracy: %.4f, Micro F1: %.4f" % (accuracy, microF1))
    print("Previous best: Accuracy: %.4f, Micro F1: %.4f" % (compare_dict['acc'], compare_dict['f1']))
    plot_2_cm(cm, compare_dict['cm'], ['angry', 'happy', 'sad', 'others'][:NUM_CLASSES], titles=['Model', 'Previous best'])

    model_compare_dict = {'f1': microF1, 'acc': accuracy, 'cm': cm}

    return y_pred, model_compare_dict