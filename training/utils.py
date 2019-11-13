"""Some helper functions.


Note: This module contains parts, which were written by  https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/utils.py
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from sklearn.metrics import confusion_matrix
from training import LOGS_DIR

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
    _, indicies = torch.max(pred_tensor, 1)
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
    valid = (label > 0)  # ignore the dominant 0 labels
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc

def compute_mIoU_and_Acc(preds, labels, numClass):
    cls_map = make_classification_map(preds)
    cls_map = cls_map.cpu().detach().numpy()
    label_map = labels.cpu().detach().numpy()
    intersection, union = intersection_and_union(cls_map, label_map, numClass)
 
    mIoU = np.mean(intersection / (union + 1e-10)) 
    acc = accuracy(cls_map, label_map)

    return mIoU, acc

def compute_confusion_matrix(preds, labels):
    '''Note: the input will be a batch
    '''
    preds = make_classification_map(preds).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    cm = confusion_matrix(preds.flatten(), labels.flatten(), labels=[0,1,2])

    return cm

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title=None,
                        cmap=plt.cm.Blues):
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix: ' + title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va="center",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    cm_dir_path = LOGS_DIR/'cm'
    if not cm_dir_path.exists():
        cm_dir_path.mkdir()
    cm_path = cm_dir_path/(str(title)+'.png')
    
    ax.set_ylim(len(cm)-0.5, -0.5)
    fig.savefig(str(cm_path), bbox_inches='tight')
    plt.close('all')
