import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# def dice_coef(y_true, y_pred):
#     intersection = np.sum(y_true * y_pred)
#     smooth = 1.0
#     return ((2. * intersection) + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
#
#
# def dice_score(y_true, y_pred, num_classes):
#     _, y_true = torch.max(y_true, 1)
#     _, y_pred = torch.max(y_pred, 1)
#     dice = []
#     for i in range(num_classes):
#         true = (y_true == i).reshape(-1).cpu().numpy()
#         pred = (y_pred == i).reshape(-1).cpu().numpy()
#         dice.append(dice_coef(true, pred))
#
#     return dice


def conf_matrix(y_true, y_pred, num_classes):
    _, y_true = torch.max(y_true, 1)
    y_true = y_true.cpu().numpy()
    _, y_pred = torch.max(y_pred, 1)
    y_pred = y_pred.cpu().numpy()
    cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1), labels=np.arange(num_classes))

    return cm


def precision(y_true, y_pred, num_classes):
    class_precisions = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = precision_score(true, pred, zero_division=1)  # * y_true.shape[0]
        class_precisions.append(value)

    return class_precisions


def sensitivity(y_true, y_pred, num_classes):
    class_sensitivities = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = recall_score(true, pred, zero_division=1)  # * y_true.shape[0]
        class_sensitivities.append(value)

    return class_sensitivities


def specificity(y_true, y_pred, num_classes):
    class_specifities = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = recall_score(true, pred, pos_label=0, zero_division=1)  # * y_true.shape[0]
        class_specifities.append(value)

    return class_specifities


def accuracy(y_true, y_pred, num_classes):
    class_accuracies = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = accuracy_score(true, pred)  # * y_true.shape[0]
        class_accuracies.append(value)

    return class_accuracies


def F1_score(y_true, y_pred, num_classes):
    class_f1_scores = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = f1_score(true, pred, zero_division=1)  # * y_true.shape[0]
        class_f1_scores.append(value)

    return class_f1_scores
