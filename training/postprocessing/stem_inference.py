import torch
from torch import nn
import numpy as np
import cv2
import warnings
from time import process_time

# try to load cpp extensions for faster inference of stem positions
CPU_OPTION_AVAILABLE, CUDA_OPTION_AVAILABLE = True, True
try:
    from stem_inference_cpu import cast_votes_cpu
except ImportError:
    warnings.warn("C++ extension for stem inference on CPU not found.")
    CPU_OPTION_AVAILABLE = False
try:
    from stem_inference_cuda import cast_votes_cuda
except ImportError:
    warnings.warn("C++/CUDA extension for stem inference on CUDA device not found.")
    CUDA_OPTION_AVAILABLE = False



class StemInference(nn.Module):
    """Infere stem positions form predicted keypoints an offsets.
    """

    def __init__(self, input_width, input_height, keypoint_radius, threshold_votes,
                 threshold_peaks, kernel_size_votes, kernel_size_peaks, device_option):
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

        assert device_option in ['naive', 'cpu', 'cuda']
        assert kernel_size_votes>0 and kernel_size_votes%2!=0
        assert kernel_size_peaks>0 and kernel_size_peaks%2!=0

        self.device_option = device_option

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
        votes_xs = torch.round(self.xs+self.keypoint_radius*stem_offset_output[:, 0])
        votes_ys = torch.round(self.ys+self.keypoint_radius*stem_offset_output[:, 1])

        votes_xs = torch.as_tensor(votes_xs, dtype=torch.long, device=votes_xs.device)
        votes_ys = torch.as_tensor(votes_ys, dtype=torch.long, device=votes_ys.device)

        votes_xs = votes_xs.detach()
        votes_ys = votes_ys.detach()
        votes_weights = stem_keypoint_output[:, 0].detach()

        if self.device_option=='naive' or (not CPU_OPTION_AVAILABLE and not CUDA_OPTION_AVAILABLE):
          votes = self.cast_votes_naive(votes_xs, votes_ys, votes_weights)
        elif self.device_option=='cpu' or not CUDA_OPTION_AVAILABLE or votes_xs.device=='cpu':
          device = votes_xs.device
          votes_xs = votes_xs.detach().cpu()
          votes_ys = votes_ys.detach().cpu()
          votes_weights = votes_weights.cpu()
          votes = cast_votes_cpu(votes_xs, votes_ys, votes_weights, self.threshold_votes).to(device)
        else:
          # cuda
          votes = cast_votes_cuda(votes_xs, votes_ys, votes_weights, self.threshold_votes)

        # normalize by size of keypoint disk
        # only relevant should we experiment with different keypoint radii
        # votes = votes/(np.pi*self.keypoint_radius*self.keypoint_radius)

        votes = self.box_filter(votes[:, None])
        votes_dilated = self.max_filter(votes)
        peaks = torch.isclose(votes, votes_dilated)&(votes>self.threshold_peaks)
        peaks = peaks[:, 0]

        # debug output
        # image = peaks.detach().cpu().numpy()
        # image = image[0]
        # image = 255*image.astype(np.uint8)
        # # image /= np.max(image)
        # cv2.imshow('votes', image)
        # cv2.waitKey()

        return [peaks[batch_index].nonzero() for batch_index in range(peaks.size(0))]


    def cast_votes_naive(self, votes_xs, votes_ys, votes_weights):
        """Naive casting of votes.
        """
        device = votes_xs.device

        # use numpy, even if we are on cuda device
        votes_xs = votes_xs.detach().cpu().numpy()
        votes_ys = votes_ys.detach().cpu().numpy()
        votes_weights = votes_weights.detach().cpu().numpy()

        batch_size = votes_xs.shape[0]

        # set weight to zero for votes that point outside image
        votes_weights = np.where(np.clip(votes_xs, 0, self.input_width-1)!=votes_xs, 0.0, votes_weights)
        votes_weights = np.where(np.clip(votes_ys, 0, self.input_height-1)!=votes_ys, 0.0, votes_weights)
        votes = np.zeros_like(votes_weights)
        for batch_index in range(batch_size):
            positions = np.where(votes_weights[batch_index]>self.threshold_votes)
            for y, x in np.transpose(positions):
                vote_x = votes_xs[batch_index, y, x]
                votes[batch_index,
                      votes_ys[batch_index, y, x],
                      votes_xs[batch_index, y, x]] += votes_weights[batch_index, y, x]

        return torch.from_numpy(votes).to(device)

