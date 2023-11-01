import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
import os
import cv2


class WFLoss(nn.Module):
    def __init__(self, gamma, weight=1.0):
        super(WFLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.LCE = CELoss()

    def forward(self, y_pred, y_true):
        '''
            y_true shape: NxCxHxW
            y_pred shape: NxCxHxW
            '''

        lce = self.LCE(y_pred, y_true)

        N = y_true.shape[0] * y_true.shape[2] * y_true.shape[3]
        loss = -self.weight * lce * torch.pow((y_pred - 1), self.gamma) * y_true * torch.log(y_pred)
        wf_loss = torch.sum(loss) / N

        return wf_loss


class FLoss(nn.Module):
    def __init__(self, gamma, weight=1.0):
        super(FLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, y_pred, y_true):
        '''
            y_true shape: NxCxHxW
            y_pred shape: NxCxHxW
            '''

        N = y_true.shape[0] * y_true.shape[2] * y_true.shape[3]
        t2 = -self.weight * torch.pow((y_pred - 1), self.gamma) * y_true * torch.log(y_pred)
        loss = torch.sum(t2) / N

        return loss


class CELoss(nn.Module):
    def __init__(self, weight=None):
        super(CELoss, self).__init__()
        self.weights = weight

    def forward(self, y_pred, y_true):
        '''
                    y_true shape: NxCxHxW
                    y_pred shape: NxCxHxW
        '''

        N = y_true.shape[0] * y_true.shape[2] * y_true.shape[3]
        loss = -torch.sum(y_true * torch.log(y_pred)) / N

        return loss


class SSLoss(nn.Module):
    def __init__(self, beta=0.1, C=0.01):
        super(SSLoss, self).__init__()
        self.beta = beta
        self.C = C
        self.LCE = CELoss()

    def forward(self, y_pred, y_true):
        '''
            y_true shape: NxCxHxW
            y_pred shape: NxCxHxW
        '''

        mean_true = torch.mean(y_true, (2, 3), keepdim=True)
        std_true = torch.std(y_true, (2, 3), keepdim=True)
        mean_pred = torch.mean(y_pred, (2, 3), keepdim=True)
        std_pred = torch.std(y_pred, (2, 3), keepdim=True)

        e1 = (y_true - mean_true + self.C) / (std_true + self.C)
        e2 = (y_pred - mean_pred + self.C) / (std_pred + self.C)
        e = torch.abs(e1 - e2)

        e_max, _ = torch.max(torch.flatten(e, start_dim=2), dim=2, keepdim=True)
        e_max = torch.unsqueeze(e_max, dim=2)
        f = (e > (self.beta * e_max)).float()

        lce = self.LCE(y_pred, y_true)

        loss = e * f * lce
        M = torch.sum(f)

        ssl_loss = torch.sum(loss)/M

        return ssl_loss


def tversky_coefficient(y_true, y_predict, smooth=1.0, beta=0.3):
    intersection = torch.sum(y_true * y_predict)
    i1 = beta * torch.sum((1-y_true) * y_predict)
    i2 = (1-beta) * torch.sum(y_true * (1-y_predict))
    return (intersection + smooth) / (intersection + i1 + i2 + smooth)


class Tversky_Loss(nn.Module):
    def __init__(self, beta):
        super(Tversky_Loss, self).__init__()
        self.beta = beta

    def forward(self, y_predict, y_true):

        tversky = 0.0
        N = y_true.shape[0]
        for i in range(y_true.shape[0]):
            tversky += (1 - tversky_coefficient(y_true[i], y_predict[i], beta=self.beta))
        loss = tversky/N

        return loss


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, y_predict, y_true):
        N = y_true.shape[0] * y_true.shape[2] * y_true.shape[3]
        product_sum = torch.sum(y_true * y_predict, dim=1)
        loss = torch.sum(torch.cos((3.1416/2) * product_sum))/N

        return loss


class FocalLogLoss(nn.Module):
    def __init__(self, gamma):
        super(FocalLogLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y_predict, y_true):
        loss = torch.ones_like(y_true)
        N = y_true.shape[0] * y_true.shape[1] * y_true.shape[2] * y_true.shape[3]
        wrong_predictions = y_predict[y_true == 0]
        loss[y_true == 0] = -15 * torch.pow(wrong_predictions, 2)
        right_predictions = y_predict[y_true == 1]
        loss[y_true != 0] = 15 * torch.pow((right_predictions - 1), self.gamma) * torch.log(right_predictions)

        loss = -torch.sum(loss)/N

        return loss


class LogMaxLoss(nn.Module):
    def __init__(self, gamma):
        super(LogMaxLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y_predict, y_true):

        y_false = 1.0 * torch.logical_not(y_true)
        loss1 = 5 * torch.pow(torch.sum(y_false * y_predict, dim=1), 2)
        loss2 = -5 * torch.sum(torch.pow((y_predict - 1), self.gamma) * y_true * torch.log(y_predict), dim=1)

        log_max_loss = torch.mean(torch.maximum(loss1, loss2))

        return log_max_loss


class PolyLogLoss(nn.Module):
    def __init__(self, gamma, weight=1.0):
        super(PolyLogLoss, self).__init__()
        self.gamma = gamma
        self.weights = weight

    def forward(self, y_predict, y_true):

        right_predictions = torch.sum(y_true * y_predict, 1)
        loss1 = -torch.pow((right_predictions - 1), self.gamma) * self.weights * torch.log(right_predictions)

        y_false = 1.0 * torch.logical_not(y_true)
        wrong_predictions = torch.sum(y_false * y_predict, 1)
        loss2 = -torch.pow(wrong_predictions, self.gamma) * torch.log(wrong_predictions)

        poly_log_loss = torch.mean(torch.abs(loss1 - loss2))

        return poly_log_loss


