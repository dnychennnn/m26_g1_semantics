"""Loss functions we use.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2


class SemanticLoss(nn.Module):
    """Cross entropy loss.

    Optional with a weight per pixel to account of objects of different sizes.
    """
    def __init__(self, weight_background, weight_weed, weight_sugar_beet, ignore_index):
        super().__init__()
        self.class_weights = torch.tensor([weight_background, weight_weed, weight_sugar_beet],
                                           dtype=torch.float32, requires_grad=False)

        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=self.class_weights, reduction='none')


    def forward(self, semantic_output_batch, semnatic_target_batch, semantic_loss_weight_batch):
        loss_batch = self.criterion(semantic_output_batch, semnatic_target_batch)

        if semantic_loss_weight_batch is not None:
            # weighted mean depending on object size
            loss_batch = torch.mean(semantic_loss_weight_batch*loss_batch)
        else:
            loss_batch = torch.mean(loss_batch)

        return loss_batch


class StemClassificationLoss(nn.Module):
    """Binary cross entropy loss to distinguish keypoint (stem) pixels from background pixels.
    """

    def __init__(self, weight_background, weight_stem):
        super().__init__()
        self.pos_weight = torch.tensor([weight_stem/(weight_stem+weight_background)],
                                       dtype=torch.float32, requires_grad=False)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)


    def forward(self, stem_keypoint_output_batch, stem_keypoint_target_batch):
        loss = self.criterion(stem_keypoint_output_batch, stem_keypoint_target_batch)

        if torch.isnan(loss):
            print('Stem classification loss: Nan encountered.')
            return torch.tensor(0.0, device=loss.device)

        return loss


class StemRegressionLoss(nn.Module):
    """Loss for x and y offset of all keypoint pixels.

    Adapted from code originally written for MGE-MSR-P-S.
    """

    def __init__(self):
        super().__init__()
        # self.tanh = nn.Tanh()
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()


    def forward(self, stem_offset_output_batch, stem_keypoint_target_batch,
                stem_offset_target_batch):
        device = stem_offset_output_batch.device

        # apply tanh to have offsets in range -1, 1
        # normalized_offset_output_batch = self.tanh(stem_offset_output_batch)

        # no tanh, use plain logits
        normalized_offset_output_batch = stem_offset_output_batch

        # make a mask containing all keypoint pixels
        # shape (batch_size, 2, target_height, target_width,)
        keypoint_mask_batch = torch.cat(2*[stem_keypoint_target_batch>0], dim=1)

        # debug output
        # np_keypoint_mask = keypoint_mask_batch.cpu().detach().numpy()
        # cv2.imshow('keypoint_mask', (225*np_keypoint_mask[0, 0]).astype(np.uint8))
        # np_offsets = stem_offset_target_batch.cpu().detach().numpy()
        # cv2.imshow('offsets', (225*np_offsets[0, 0]).astype(np.uint8))
        # cv2.waitKey()

        if not torch.any(keypoint_mask_batch):
            print('Stem regression loss: No pixels in mask.')
            return torch.tensor(0.0, device=device)

        # get offsets in mask from output and target
        masked_offset_output = normalized_offset_output_batch[keypoint_mask_batch]
        masked_offset_target = stem_offset_target_batch[keypoint_mask_batch]

        # debug output
        # print(torch.min(masked_offset_output))
        # print(torch.max(masked_offset_output))
        # print(torch.min(masked_offset_target))
        # print(torch.max(masked_offset_target))

        loss = self.criterion(masked_offset_output, masked_offset_target)

        if torch.isnan(loss):
            print('Stem regression loss: Nan encountered.')
            return torch.tensor(0.0, device=device)

        return loss

