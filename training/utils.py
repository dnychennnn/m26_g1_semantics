"""Some helper functions.
"""

from matplotlib import pyplot as plt
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

