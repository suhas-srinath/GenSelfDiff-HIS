import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff

start = 1


def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    smooth = 1.0
    return ((2. * intersection) + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def dice_score(y_true, y_pred, num_classes):
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    dice = []
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        dice.append(dice_coef(true, pred))

    return dice


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
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = precision_score(true, pred, zero_division=1)  # * y_true.shape[0]
        class_precisions.append(value)

    return class_precisions


def sensitivity(y_true, y_pred, num_classes):
    class_sensitivities = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = recall_score(true, pred, zero_division=1)  # * y_true.shape[0]
        class_sensitivities.append(value)

    return class_sensitivities


def specificity(y_true, y_pred, num_classes):
    class_specifities = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = recall_score(true, pred, pos_label=0, zero_division=1)  # * y_true.shape[0]
        class_specifities.append(value)

    return class_specifities


def accuracy(y_true, y_pred, num_classes):
    class_accuracies = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = accuracy_score(true, pred)  # * y_true.shape[0]
        class_accuracies.append(value)

    return class_accuracies


def F1_score(y_true, y_pred, num_classes):
    class_f1_scores = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = f1_score(true, pred, zero_division=1)  # * y_true.shape[0]
        class_f1_scores.append(value)

    return class_f1_scores


def Jaccard_score(y_true, y_pred, num_classes):
    class_jaccard_scores = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = jaccard_score(true, pred, zero_division=1)  # * y_true.shape[0]
        class_jaccard_scores.append(value)

    return class_jaccard_scores


def Hausdorff_distance(y_true, y_pred, num_classes):
    class_hd = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(start, num_classes):
        true = (y_true == i).squeeze(0).cpu().numpy()
        pred = (y_pred == i).squeeze(0).cpu().numpy()
        hd1 = directed_hausdorff(true, pred)[0]
        hd2 = directed_hausdorff(pred, true)[0]
        hd = max(hd1, hd2)
        class_hd.append(hd)

    return class_hd

# This script is adopted from the offical repository of AJI score
def Aggregated_jaccard_index(gt_map, predicted_map, gpu):
    _, gt_map = torch.max(gt_map, 1)
    _, predicted_map = torch.max(predicted_map, 1)

    gt_list = torch.unique(gt_map)
    pr_list = torch.unique(predicted_map)

    if start != 0:
        gt_list = gt_list[gt_list != 0]
        pr_list = pr_list[pr_list != 0]

    pr_list = torch.cat((pr_list.view(-1, 1), torch.zeros(pr_list.size(0), 1).to(gpu)), dim=1)

    overall_correct_count = 0.0
    union_pixel_count = 0.0

    i = len(gt_list)

    while len(gt_list) > 0:
        # print(f'Processing object # {i}')

        gt = (gt_map == gt_list[i - 1]).float()

        predicted_match = gt * predicted_map.float()

        if predicted_match.sum() == 0:
            union_pixel_count += gt.sum()
            gt_list = gt_list[:-1]
            i = len(gt_list)
        else:
            predicted_nuc_index = torch.unique(predicted_match)
            if start != 0:
                predicted_nuc_index = predicted_nuc_index[predicted_nuc_index != 0]

            JI = 0
            best_match = None

            for j in range(len(predicted_nuc_index)):
                matched = (predicted_map == predicted_nuc_index[j]).float()
                nJI = matched.logical_and(gt).sum() / matched.logical_or(gt).sum()

                if nJI > JI:
                    best_match = predicted_nuc_index[j]
                    JI = nJI

            predicted_nuclei = (predicted_map == best_match).float()

            overall_correct_count += (gt.logical_and(predicted_nuclei)).sum()
            union_pixel_count += (gt.logical_or(predicted_nuclei)).sum()

            gt_list = gt_list[:-1]
            i = len(gt_list)

            best_match_idx = (pr_list[:, 0] == best_match).nonzero().item()
            pr_list[best_match_idx, 1] += 1

    unused_nuclei_list = (pr_list[:, 1] == 0).nonzero().view(-1)

    for k in range(len(unused_nuclei_list)):
        print(pr_list[unused_nuclei_list[k], 0])
        unused_nuclei = (predicted_map == pr_list[unused_nuclei_list[k], 0]).float()
        union_pixel_count += unused_nuclei.sum()

    if overall_correct_count == 0 and union_pixel_count == 0:
        return 1.0
    aji = overall_correct_count / union_pixel_count

    return aji.cpu().numpy()



