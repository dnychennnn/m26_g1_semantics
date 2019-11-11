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

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

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
    return acc, valid_sum
