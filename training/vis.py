"""Make nice images from tensors.
"""

import torch
from torch import nn
import numpy as np
import cv2


def tensor_to_bgr(tensor, mean_rgb=None, std_rgb=None):
    """Convert tensor to numpy.array with BGR channels last.

    Optionally undo normalization with the provided parameters.

    If tensor is batch, only pick the first slice.

    Args:
        tensor (torch.Tensor): Tensor to convert.
        mean_rgb (list): Mean of RGB used for normalization.
        std_rgb (list): Standard deviation of RGB used for normalization.

    Returns:
        numpy.array of shape (height, width, 3,).
    """
    while len(tensor.shape)>3:
        tensor = tensor[0]

    assert len(tensor.shape)==3

    tensor = tensor.cpu().detach().numpy()
    tensor = tensor.transpose((1, 2, 0))[..., :3]

    if mean_rgb is not None and std_rgb is not None:
        tensor *= np.array(std_rgb).reshape(1, 1, 3)
        tensor += np.array(mean_rgb).reshape(1, 1, 3)

    # rgb to bgr
    tensor = tensor[..., ::-1]

    return tensor


def tensor_to_single_channel(tensor, mean=None, std=None):
    """Convert single channel tensor to numpy.array.

    Optionally undo normalization with the provided parameters.

    If tensor is batch, only pick the first slice.

    Args:
        tensor (torch.Tensor): Tensor to convert.
        mean (float): Mean used for normalization.
        std (float): Standard deviation used for normalization.

    Return:
        numpy.array of shape (height, width,).
    """
    while len(tensor.shape)>2:
        tensor = tensor[0]

    assert len(tensor.shape)==2

    tensor = tensor.cpu().detach().numpy()

    if mean is not None and std is not None:
        tensor *= std
        tensor += mean

    return tensor


def tensor_to_false_color(tensor_rgb, tensor_nir, mean_rgb, std_rgb, mean_nir, std_nir):
    """False color image by replacing green channel with NIR.
    """
    image_bgr = tensor_to_bgr(tensor_rgb, mean_rgb=mean_rgb, std_rgb=std_rgb)
    image_nir = tensor_to_single_channel(tensor_nir, mean=mean_nir, std=std_nir)

    # replace green channel with NIR
    image_bgr[..., 1] = image_nir

    return image_bgr



def make_plot_from_semantic_output(input_rgb, input_nir, semantic_output, apply_softmax, **normalization):
    image = tensor_to_false_color(input_rgb, input_nir, **normalization)

    # use grayscale as background
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = np.stack(3*[gray], axis=-1)

    # show sugar beet heatmap in blue
    sugar_beet_color = np.array([1.0, 0.0, 0.0]).reshape(1, 1, 3)

    # show weed heatmap in yellow
    weed_color = np.array([0.0, 1.0, 1.0]).reshape(1, 1, 3)

    while len(semantic_output.shape)>3:
        semantic_output = semantic_output[0]

    if apply_softmax:
        semantic_output = nn.functional.softmax(semantic_output, dim=0)

    weed_heatmap = semantic_output[1, :, :].detach().cpu().numpy()[..., None]
    sugar_beet_heatmap = semantic_output[2, :, :].detach().cpu().numpy()[..., None]

    plot = background+0.5*weed_color*weed_heatmap+0.5*sugar_beet_color*sugar_beet_heatmap
    plot = np.clip(plot, 0.0, 1.0)

    return plot


def make_plot_from_stem_output(input_rgb, input_nir, stem_keypoint_output, stem_offset_output, keypoint_radius, apply_sigmoid, apply_tanh, **normalization):
    """
    Note: This function contains parts, which were written for other student projects
    conducted by the author.
    """
    image = tensor_to_false_color(input_rgb, input_nir, **normalization)

    height, width = image.shape[:2]

    # use grayscale as background
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = np.stack(3*[gray], axis=-1)

    while len(stem_keypoint_output.shape)>2:
        stem_keypoint_output = stem_keypoint_output[0]

    while len(stem_offset_output.shape)>3:
        stem_offset_output = stem_offset_output[0]

    if apply_sigmoid:
        stem_keypoint_output = torch.sigmoid(stem_keypoint_output)

    if apply_tanh:
        stem_offset_output = torch.tanh(stem_offset_output)

    stem_heatmap = stem_keypoint_output.detach().cpu().numpy()[..., None]
    stem_offset_x = stem_offset_output[0]
    stem_offset_y = stem_offset_output[1]

    # show stem heatmap in blue
    stem_color = np.array([1.0, 0.0, 0.0]).reshape(1, 1, 3)

    plot = background+0.5*stem_color*stem_heatmap

    # show offsets as tiny arrows
    arrows = np.zeros((height, width, 3,), dtype=np.float)

    grid_distance = 10
    for x in range(0, width, grid_distance):
        for y in range(0, height, grid_distance):
            if stem_heatmap[y, x]>0.3:
                offset_x = keypoint_radius*stem_offset_x[y, x]
                offset_y = keypoint_radius*stem_offset_y[y, x]
                cv2.arrowedLine(arrows, (x, y), (x+offset_x, y+offset_y), (1.0, 1.0, 1.0), thickness=1, tipLength=0.2)

    plot = plot+0.5*arrows
    plot = np.clip(plot, 0.0, 1.0)

    return plot

