"""Some helper functions.


Note: This module contains parts, which were written by  https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/utils.py
"""

from matplotlib import pyplot as plot
import numpy as np
import cv2
import torch


def visualize(image, mask):
    plot.figure(1)
    plot.subplot(211)
    plot.imshow(image)

    plot.subplot(212)
    plot.imshow(mask)
    plot.show()

def get_confidence_map(confidence):
    min_confidence = np.min(confidence)
    max_confidence = np.max(confidence)
    print('confidence min, max =', min_confidence, max_confidence)

    confidence_map = (255.0*(confidence-min_confidence/(max_confidence-min_confidence+0.0001))).astype(np.uint8)
    return confidence_map


def visualize_single_confidence(cv_input, cv_target, cv_confidence):
    cv2.imshow('taget sugar beet', cv_target)
    cv2.imshow('input', cv_input)
    cv2.imshow('sugar beet', cv_confidence)
    cv2.waitKey(1)

def make_classification_map(pred_tensor):
    _, indicies = torch.max(pred_tensor, 0)
    return indicies

def intersection_and_union(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc

def compute_IoU_and_Acc(preds, labels):
    cls_map = make_classification_map(preds)
    cls_map = cls_map.cpu().detach().numpy()
    label_map = labels.cpu().detach().numpy()
    intersection, union = intersection_and_union(cls_map, label_map, 3)
    IoU = np.sum(intersection) / np.sum(union)
    acc = accuracy(cls_map, label_map)

    return IoU, acc

