"""Final output layers for our network.

Author: Jan Quakernack
"""

import torch
from torch import nn
import numpy as np
import cv2

from postprocessing_modules_cpp import cast_votes_cpp

# inference of stem positions from network output is done in C++ and interfaced here

class StemVoting(nn.Module):
    '''Each pixels casts a vote using the predicted offset weighted by the keypoint confidence.
    '''

    def __init__(self, input_width, input_height, keypoint_radius):
        super().__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.keypoint_radius = keypoint_radius

        # out a grid of x and y coordinates in the buffer, which we will need later
        self.register_buffer('xs', torch.arange(input_width, dtype=torch.float32).reshape(1, -1))
        self.register_buffer('ys', torch.arange(input_height, dtype=torch.float32).reshape(-1, 1))

        self.img_small_width = self.input_width
        self.img_small_height = self.input_height
        self.img_width = self.input_width
        self.img_height = self.input_height


    def forward(self, stem_keypoint_output, stem_offset_output):
        votes_xs = torch.round(self.xs+self.keypoint_radius*stem_offset_output[:, 0])
        votes_ys = torch.round(self.ys+self.keypoint_radius*stem_offset_output[:, 1])

        votes_xs = torch.as_tensor(votes_xs, dtype=torch.long, device=votes_xs.device)
        votes_ys = torch.as_tensor(votes_ys, dtype=torch.long, device=votes_ys.device)

        votes_xs = votes_xs.detach().cpu()
        votes_ys = votes_ys.detach().cpu()
        votes_weights = stem_keypoint_output[:, 0].detach().cpu()

        votes = cast_votes_cpp(votes_xs, votes_ys, votes_weights, 0.01)

        image = votes.detach().cpu().numpy()
        image = image[0]
        image /= np.max(image)
        cv2.imshow('image', image)
        cv2.waitKey()

        exit()
