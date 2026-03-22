import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,f1_score,confusion_matrix, roc_auc_score,roc_curve, auc)
from sklearn.preprocessing import label_binarize
import itertools

# BASIC METRICS


def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


# CONFUSION MATRIX


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return cm


# ROC CURVE & AUC


def compute_auc(y_true, y_pred_probs, num_classes):
    # Convert labels to one-hot
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    return roc_auc_score(y_true_bin, y_pred_probs, average='weighted', multi_class='ovr')


def plot_roc_multiclass(y_true, y_pred_probs, num_classes, class_names):
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multiclass)")
    plt.legend()
    plt.show()

    return roc_auc