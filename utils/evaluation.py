#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 20:23
# @Author  : Denxun
# @FileName: evaluation.py
# @Software: PyCharm
import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def segmentation_recall(predicted, target):
    smooth =0.1
    def threshold_tensor(tensor, threshold):
        #tensor[tensor > threshold] = 1
        tensor[tensor <= threshold] = 0
        return tensor
    # 示例输入张量
    input_tensor = predicted
    # 设置阈值为 0.5
    threshold = 0.5
    # 应用阈值处理
    output_tensor = threshold_tensor(input_tensor, threshold)

    intersection = (output_tensor * target).sum(2).sum(2).sum(1) + smooth
    target_sum = target.sum(2).sum(2).sum(1)
    recall = intersection / (target_sum + smooth)
    return recall.mean()

def segmentation_precision(predicted, target):
    smooth =0.1
    def threshold_tensor(tensor, threshold):
        #tensor[tensor > threshold] = 1
        tensor[tensor <= threshold] = 0
        return tensor
    # 示例输入张量
    input_tensor = predicted
    # 设置阈值为 0.5
    threshold = 0.5
    # 应用阈值处理
    output_tensor = threshold_tensor(input_tensor, threshold)

    intersection = (output_tensor * target).sum(2).sum(2).sum(1) + smooth
    precision = intersection / (output_tensor.sum(2).sum(2).sum(1) + smooth)
    return precision.mean()
class binary_miou(nn.Module):
    def __init__(self, n_classes=2):
        super(binary_miou, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _iou_score(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        score = (intersect + smooth) / (z_sum + y_sum-intersect + smooth)
        score = score
        return score

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        #assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        Iou = 0.0
        input_tensor1=1-inputs
        input_tensor2=inputs
        pred=torch.cat([input_tensor1,input_tensor2],dim=1)
        for i in range(0, self.n_classes):
            dice = self._iou_score(pred[:, i], target[:, i])
            class_wise_dice.append(dice.item())
            Iou += dice * weight[i]
        return Iou / self.n_classes

class binary_Dice(nn.Module):
    def __init__(self, n_classes=2):
        super(binary_Dice, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _iou_score(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        score = (2*intersect + smooth) / (z_sum + y_sum + smooth)
        score = score
        return score

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        #assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        dice = 0.0
        input_tensor1=1-inputs
        input_tensor2=inputs
        pred=torch.cat([input_tensor1,input_tensor2],dim=1)
        for i in range(0, self.n_classes):
            dice = self._iou_score(pred[:, i], target[:, i])
            class_wise_dice.append(dice.item())
            dice += dice * weight[i]
        return dice / self.n_classes
def segmentation_f1_score(precision, recall):
    # precision = segmentation_precision(predicted, target)
    # recall = segmentation_recall(predicted, target)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)  # 添加一个很小的值以防止除零错误
    return f1_score
def segmentation_mae(predicted, target):
    return torch.abs(predicted - target).mean()

def false_positive_rate(predicted, target):
    smooth = 0.000001  # 平滑因子，用于避免分母为零
    def threshold_tensor(tensor, threshold):
        tensor[tensor > threshold] = 1
        tensor[tensor <= threshold] = 0
        return tensor
    input_tensor = predicted
    threshold = 0.5
    predicted = threshold_tensor(input_tensor, threshold)
    # intersection = (output_tensor * target).sum(2).sum(2).sum(1) + smooth
    # target_sum = target.sum(2).sum(2).sum(1)
    false_positives = ((predicted == 1) & (target == 0)).sum(2).sum(2).sum(1)
    true_negatives = ((predicted == 0) & (target == 0)).sum(2).sum(2).sum(1)
    fpr = (false_positives+smooth )/ (false_positives + true_negatives+smooth)
    return fpr.mean()

def dice_coefficient(predicted, target):
    smooth = 0.1
    def threshold_tensor(tensor, threshold):
        #tensor[tensor > threshold] = 1
        tensor[tensor <= threshold] = 0
        return tensor
    # 示例输入张量
    input_tensor = predicted
    # 设置阈值为 0.5
    threshold = 0.5
    # 应用阈值处理
    output_tensor = threshold_tensor(input_tensor, threshold)
    predicted=output_tensor
    product = predicted * target
    #print(product)
    #print(product.shape)
    intersection = 2*(product.sum(2).sum(2).sum(1) + smooth)
    # print(intersection)
    # print(intersection.shape)
    union = predicted.sum(2).sum(2).sum(1) + target.sum(2).sum(2).sum(1) + smooth
    return (intersection / union).mean()
def tp_fp_tn_fn(predicted, target):
    smooth = 0.1
    def threshold_tensor(tensor, threshold):
        tensor[tensor > threshold] = 1
        tensor[tensor <= threshold] = 0
        return tensor
    input_tensor = predicted
    threshold = 0.5
    #output_tensor = threshold_tensor(input_tensor, threshold)

    target=np.array(target,dtype=np.float32)
    print(target)
    output_tensor=np.array(input_tensor,dtype=np.float32)
    pred_positive = output_tensor#[:, 0, :, :]
    #pred_negative = output_tensor[:, 1, :, :]
    y_pred_binary = np.where(pred_positive > 0.5, 1, 0)
    y_true_flat = target.flatten()
    y_pred_flat = y_pred_binary.flatten()
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    tp = cm[1, 1]
    fp = cm[0, 1]
    tn = cm[0, 0]
    fn = cm[1, 0]
    return tp.mean(),fp.mean(),tn.mean(),fn.mean()

def iou(predicted, target):
    smooth = 0.1
    def threshold_tensor(tensor, threshold):
        #tensor[tensor > threshold] = 1
        tensor[tensor <= threshold] = 0
        return tensor
    input_tensor = predicted
    threshold = 0.5
    output_tensor = threshold_tensor(input_tensor, threshold)
    predicted=output_tensor
    intersection = (predicted * target).sum(2).sum(2).sum(1)
    union = predicted.sum(2).sum(2).sum(1) + target.sum(2).sum(2).sum(1) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()