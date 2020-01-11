import torch
import cv2
import numpy as np


def make_classification_map(semantic_output_batch, sugar_beet_threshold, weed_threshold):
    """Select the class with maximum confidence.

    Args:
        semantic_output_batch (torch.Tensor): Class confidences of shape (batch_size, num_classes, height, width,).
        sugar_beet_threshold (float): Confidence threshold.
        weed_threshold (float): Confidence threshold.

    Returns:
        A torch.tensor of shape (batch_size, height, width,) with integer class labels.
    """

    batch_size, _, height, width = semantic_output_batch.shape

    # background by default, so initialize with zeros
    semantic_labels = torch.zeros((batch_size, height, width),
            dtype=torch.int, device=semantic_output_batch.device)

    mask_weed = semantic_output_batch[:, 1]>weed_threshold
    mask_sugar_beet = semantic_output_batch[:, 2]>sugar_beet_threshold

    semantic_labels[mask_weed] = 1
    semantic_labels[mask_sugar_beet] = 2

    return semantic_labels
