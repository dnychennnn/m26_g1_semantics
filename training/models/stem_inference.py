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

    def __init__(self, input_width, input_height, keypoint_radius, threshold, device_option='cuda'):
        """
        Args:
            threshold (float): Pixel with a keypoint confidence below this value are not used for stem inference.
        """
        super().__init__()

        assert device_option in ['naive', 'cpu', 'cuda']
        self.device_option = device_option

        self.input_width = input_width
        self.input_height = input_height
        self.keypoint_radius = keypoint_radius
        self.threshold = threshold

        # put a grid of x and y coordinates in the buffer, which we will need later
        self.register_buffer('xs', torch.arange(input_width, dtype=torch.float32).reshape(1, -1))
        self.register_buffer('ys', torch.arange(input_height, dtype=torch.float32).reshape(-1, 1))


    def forward(self, stem_keypoint_output, stem_offset_output):
        votes_xs = torch.round(self.xs+self.keypoint_radius*stem_offset_output[:, 0])
        votes_ys = torch.round(self.ys+self.keypoint_radius*stem_offset_output[:, 1])

        votes_xs = torch.as_tensor(votes_xs, dtype=torch.long, device=votes_xs.device)
        votes_ys = torch.as_tensor(votes_ys, dtype=torch.long, device=votes_ys.device)

        votes_xs = votes_xs.detach()
        votes_ys = votes_ys.detach()
        votes_weights = stem_keypoint_output[:, 0].detach()

        if self.device_option=='naive' or (not CPU_OPTION_AVAILABLE and not CUDA_OPTION_AVAILABLE):
          device = votes_xs.device
          votes_xs = votes_xs.cpu()
          votes_ys = votes_ys.cpu()
          votes_weights = votes_weights.cpu()
          votes = self.cast_votes_naive(votes_xs, votes_ys, votes_weights).to(device)
        elif self.device_option=='cpu' or not CUDA_OPTION_AVAILABLE or votes_xs.device=='cpu':
          device = votes_xs.device
          votes_xs = votes_xs.cpu()
          votes_ys = votes_ys.cpu()
          votes_weights = votes_weights.cpu()
          votes = cast_votes_cpu(votes_xs, votes_ys, votes_weights, self.threshold).to(device)
        else:
          # cuda
          votes = cast_votes_cuda(votes_xs, votes_ys, votes_weights, self.threshold)

        image = votes.detach().cpu().numpy()
        image = image[0]
        image /= np.max(image)
        cv2.imshow('votes', image)
        cv2.waitKey()

        return votes


    def cast_votes_naive(self, votes_xs, votes_ys, votes_weights):
        """Naive casting of votes.
        """
        # use numpy, even if we are on cuda device
        votes_xs = votes_xs.numpy()
        votes_ys = votes_ys.numpy()
        votes_weights = votes_weights.numpy()

        batch_size = votes_xs.shape[0]

        # set weight to zero for votes that point outside image
        votes_weights = np.where(np.clip(votes_xs, 0, self.input_width-1)!=votes_xs, 0.0, votes_weights)
        votes_weights = np.where(np.clip(votes_ys, 0, self.input_height-1)!=votes_ys, 0.0, votes_weights)
        votes = np.zeros_like(votes_weights)
        for batch_index in range(batch_size):
            positions = np.where(votes_weights[batch_index]>self.threshold)
            for y, x in np.transpose(positions):
                vote_x = votes_xs[batch_index, y, x]
                votes[batch_index,
                      votes_ys[batch_index, y, x],
                      votes_xs[batch_index, y, x]] += votes_weights[batch_index, y, x]

        return torch.from_numpy(votes)

