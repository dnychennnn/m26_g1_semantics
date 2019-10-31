"""Loss functions we use.

Author:
    Jan Quakernack
"""

import torch
import torch.nn as nn


class StemClassificationLoss(nn.Module):
    """Binary cross entropy loss to distinguish keypoint (stem) pixels
    from background pixels.
    """

    def __init__(self, weight_background, weight_stem):
        super().__init__()
        self.pos_weight = torch.tensor([weight_stem/(weight_stem+weight_background)],
                                       dtype=torch.float32, requires_grad=False)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)


    def forward(self, stem_keypoint_output_batch, stem_keypoint_target_batch):
        return self.criterion(stem_keypoint_output_batch, stem_keypoint_target_batch)


class StemRegressionLoss(nn.Module):
    """Mean squared error applied to x and y offset of all keypoint pixels.
    """

    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.criterion = nn.MSELoss()


    def forward(self, stem_offset_output_batch, stem_keypoint_target_batch,
                stem_offset_target_batch):
        # apply tanh to have offsets in range -1, 1
        normalized_offset_output_batch = self.tanh(stem_offset_output_batch)

        # make a mask containing all keypoint pixels
        # shape (batch_size, 2, target_height, target_width,)
        keypoint_mask_batch = torch.stack(2*[stem_keypoint_target_batch>0], dim=1)

        # get offsets in mask from output and target
        masked_offset_output = normalized_offset_output_batch[keypoint_mask_batch]
        masked_offset_target = stem_offset_target_batch[keypoint_mask_batch]

        return self.criterion(masked_offset_output, masked_offset_target)

