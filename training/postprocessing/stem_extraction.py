import torch
from torch import nn
import numpy as np
import cv2
import warnings
from time import process_time


class StemExtraction(nn.Module):
    """Infere stem positions form predicted keypoints an offsets.

    Adapted from code originally written for MGE-MSR-P-S.
    """

    def __init__(self, input_width, input_height, keypoint_radius, threshold_votes,
                 threshold_peaks, kernel_size_votes, kernel_size_peaks):
        """
        Args:
            threshold_votes (float): Pixels with a keypoint confidence below this value
            do not vote for the stem position.
            threshold_peaks (float): Pixels with accumulated votes below this value
            are not considered as stems.

        Forward:
            A list with positons of shape (num_stems, 2,), datatype torch.long for each slice in batch.
        """
        super().__init__()

        assert kernel_size_votes>0 and kernel_size_votes%2!=0
        assert kernel_size_peaks>0 and kernel_size_peaks%2!=0

        self.input_width = input_width
        self.input_height = input_height
        self.keypoint_radius = keypoint_radius
        self.threshold_votes = threshold_votes
        self.threshold_peaks = threshold_peaks

        # put a grid of x and y coordinates in the buffer, which we will need later
        self.register_buffer('xs', torch.arange(input_width, dtype=torch.float32).reshape(1, -1))
        self.register_buffer('ys', torch.arange(input_height, dtype=torch.float32).reshape(-1, 1))

        # run a box filted on the casted votes so each pixel also votes for some neighbors
        self.box_filter = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=kernel_size_votes,
                                    stride=1,
                                    padding=(kernel_size_votes-1)//2)
        self.box_filter.weight.data.fill_(1.0)
        self.box_filter.bias.data.zero_()

        self.max_filter = torch.nn.MaxPool2d(kernel_size=kernel_size_peaks, stride=1, padding=(kernel_size_peaks-1)//2)

        for parameter in self.parameters():
            # no need to compute the gradient for all parts of this module
            parameter.requires_grad = False


    def forward(self, stem_keypoint_output, stem_offset_output):
        batch_size = stem_keypoint_output.shape[0]

        votes_xs = torch.round(self.xs+self.keypoint_radius*stem_offset_output[:, 0])
        votes_ys = torch.round(self.ys+self.keypoint_radius*stem_offset_output[:, 1])

        votes_xs = torch.as_tensor(votes_xs, dtype=torch.long, device=votes_xs.device)
        votes_ys = torch.as_tensor(votes_ys, dtype=torch.long, device=votes_ys.device)

        votes_xs = votes_xs.detach()
        votes_ys = votes_ys.detach()
        votes_weights = stem_keypoint_output[:, 0].detach()

        votes = self.cast_votes_pytorch(votes_xs, votes_ys, votes_weights)

        # normalize by size of keypoint disk
        votes /= np.pi*self.keypoint_radius*self.keypoint_radius

        votes = self.box_filter(votes[:, None])
        votes_dilated = self.max_filter(votes)

        peaks = torch.isclose(votes, votes_dilated)&(votes>self.threshold_peaks)
        peaks = peaks[:, 0]

        positions_batch = []

        for slice_index in range(batch_size):
            positions = peaks[slice_index].nonzero()
            scores = votes[slice_index, 0][peaks[slice_index]]
            # put as (x, y, score)
            positions = torch.stack([positions[:, 1].float(), positions[:, 0].float(), scores], dim=-1)
            positions_batch.append(positions)

        return positions_batch


    def cast_votes_pytorch(self, votes_xs, votes_ys, votes_weights):
        batch_size = votes_weights.shape[0]
        device = votes_weights.device

        # make sure we only vote inside image
        votes_weights[votes_xs<0] = 0.0
        votes_weights[votes_xs>=self.input_width] = 0.0

        votes_weights[votes_ys<0] = 0.0
        votes_weights[votes_ys>=self.input_height] = 0.0

        votes_xs = torch.clamp(votes_xs, 0, self.input_width-1)
        votes_ys = torch.clamp(votes_ys, 0, self.input_height-1)

        # let only pixels above threshold vote
        votes_weights[votes_weights<self.threshold_votes] = 0.0

        # initialize accumulated votes with zeros
        votes = torch.zeros((batch_size, self.input_height, self.input_width,), dtype=torch.float32, device=device)

        # additional to x, y we also need the slice index for each vote
        votes_slices_indices = torch.stack([torch.full((self.input_height, self.input_width,),
            slice_index, dtype=torch.long, device=device) for slice_index in range(batch_size)], dim=0)

        # accumulate votes
        votes.index_put_((votes_slices_indices, votes_ys, votes_xs), votes_weights, accumulate=True)

        return votes

