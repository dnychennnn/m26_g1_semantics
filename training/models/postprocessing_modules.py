"""Final output layers for our network.

Author: Jan Quakernack
"""

import torch
from torch import nn
import numpy as np
import cv2

import stem_voting_cpp


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

        votes_xs = torch.as_tensor(votes_xs, dtype=torch.long, device=votes_xs.device).clamp(0, self.input_width-1)
        votes_ys = torch.as_tensor(votes_ys, dtype=torch.long, device=votes_ys.device).clamp(0, self.input_height-1)
        # clamping will cause some wrong vote casts at the border, but this is okay

        # print(votes_xs.shape)
        # print(votes_ys.shape)
        # print(stem_keypoint_output[:, 0].shape)

        stem_voting_cpp.cast_votes(votes_xs, votes_ys, stem_keypoint_output[:, 0])

        # score = torch.zeros((self.input_height, self.input_width,), dtype=torch.float32)
        # for x in range(self.input_width):
            # for y in range(self.input_height):
                # voting_x = votings_xs[0, y, x]
                # voting_y = votings_ys[0, y, x]
                # score[voting_y, voting_x] += stem_keypoint_output[0, voting_y, voting_x]

        # image = score.detach().cpu().numpy()
        # # image = image[0, 0]
        # image /= np.max(image)
        # print(np.min(image))
        # cv2.imshow('image', image)
        # cv2.waitKey()

        exit()
