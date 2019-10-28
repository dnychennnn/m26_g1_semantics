"""
Author:
"""

import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import cv2

from training import load_config, DATA_DIR

from torch.utils.data import Dataset
from torchvision import transforms


class SugarBeetDataset(Dataset):

    @classmethod
    def from_config(cls):
        config = load_config('sugar_beet_dataset.yaml')
        return SugarBeetDataset(**config)


    def __init__(self,
                 label_background, label_ignore, label_weed, label_sugar_beet,
                 input_height, input_width, target_height, target_width,
                 path_to_rgb_images, suffix_rgb_images,
                 path_to_nir_images, suffix_nir_images,
                 path_to_semantic_labels, suffix_semantic_labels,
                 path_to_stem_labels, suffix_stem_labels):
        """Constructor.

        Args:
            path_to_xxx (str), suffix_xxx (str): Where to find files and suffix
            specific for these files, e.g. 'annotations/dlp/iMapCleaned/',
            '_GroundTruthg_iMap.png'.
        """
        super().__init__()

        self.label_background = label_background
        self.label_ignore = label_ignore
        self.label_weed = label_weed
        self.label_sugar_beet = label_sugar_beet

        self.path_to_rgb_images = SugarBeetDataset._assure_is_absolute_data_path(Path(path_to_rgb_images))
        self.suffix_rgb_images = suffix_rgb_images

        self.path_to_nir_images = SugarBeetDataset._assure_is_absolute_data_path(Path(path_to_nir_images))
        self.suffix_nir_images = suffix_nir_images

        self.path_to_semantic_labels = SugarBeetDataset._assure_is_absolute_data_path(Path(path_to_semantic_labels))
        self.suffix_semantic_labels = suffix_semantic_labels

        self.path_to_stem_labels = SugarBeetDataset._assure_is_absolute_data_path(Path(path_to_stem_labels))
        self.suffix_stem_labels = suffix_stem_labels

        # transformations
        self.resize_input = transforms.Resize((input_height, input_width), interpolation=Image.BILINEAR)
        self.resize_target = transforms.Resize((target_height, target_width), interpolation=Image.NEAREST)

        # TODO random transformations for data augmentation

        self.pil_to_tensor = transforms.ToTensor()

        # get filenames without suffix of all items in dataset, sort them
        self.filenames = list(sorted(self._collect_filenames()))


    @classmethod
    def _assure_is_absolute_data_path(cls, path):
        """If the provided path is relative, change it to DATA_DIR/path.
        """
        return path if path.is_absolute() else DATA_DIR/path


    def _collect_filenames(self):
        """Get filenames without suffix of all items in dataset for which we
        have all four files (RGB image, NIR image, semantic label and
        stem label).
        """
        # get all filenames of RGB images without suffix
        filenames = [path.name.replace(self.suffix_rgb_images, '')
                     for path in self.path_to_rgb_images.glob('*'+self.suffix_rgb_images)]

        # filter out those for which the corresponding NIR image, semantic label
        # or stem label does not exists
        filenames = [name for name in filenames
                     if (self.get_path_to_nir_image(name).exists()
                         and self.get_path_to_semantic_label(name).exists()
                         and self.get_path_to_stem_label(name).exists())]

        return filenames


    def get_path_to_rgb_image(self, filename):
        """Construct path to RGB image in this dataset from filename without suffix.
        """
        return self.path_to_rgb_images/(filename+self.suffix_rgb_images)


    def get_path_to_nir_image(self, filename):
        """Construct path to NIR image in this dataset from filename without suffix.
        """
        return self.path_to_nir_images/(filename+self.suffix_nir_images)


    def get_path_to_semantic_label(self, filename):
        """Construct path to semantic in this dataset from filename without suffix.
        """
        return self.path_to_semantic_labels/(filename+self.suffix_semantic_labels)


    def get_path_to_stem_label(self, filename):
        """Construct path to stem label in this dataset from filename without suffix.
        """
        return self.path_to_stem_labels/(filename+self.suffix_stem_labels)


    def __getitem__(self, index):
        """Load files for the given index from dataset and convert to tensors.

        Returns:
            Tuple of RGB+NIR input tensor, semantic target.
        """
        filename = self.filenames[index]

        # load images and labels
        rgb_image = Image.open(self.get_path_to_rgb_image(filename)).convert('RGB')
        nir_image = Image.open(self.get_path_to_nir_image(filename)).convert('L')

        rgb_image = self.resize_input(rgb_image)
        nir_image = self.resize_input(nir_image)

        # no conversion to RGB here
        semantic_label = Image.open(self.get_path_to_semantic_label(filename))
        semantic_label = self.resize_target(semantic_label)
        semantic_label = np.asarray(semantic_label).astype(np.int)

        # make semantic target
        semantic_target = self._make_semantic_target(semantic_label)

        # make stem target
        # TODO

        # random transformations
        # TODO keep in mind that we need to be able to transform stem positions (x, y) in the same way

        # normalization
        # TODO

        # convert input images to tensors
        rgb_tensor = self.pil_to_tensor(rgb_image) # shape (3, input_height, input_width,)
        nir_tensor = self.pil_to_tensor(nir_image) # shape (1, input_height, input_width,)

        # concatenate RGB and NIR to single input tensor
        input_tensor = torch.cat([rgb_tensor, nir_tensor], dim=0)

        # convert targets to tensors
        semantic_target_tensor = torch.from_numpy(semantic_target) # shape (target_height, target_width,)

        return input_tensor, semantic_target_tensor


    def _make_semantic_target(self, semantic_label):
        """Remap class labels.
        """
        semantic_target = np.zeros_like(semantic_label)
        semantic_target[np.where(semantic_label)==self.label_ignore] = 1
        semantic_target[np.where(semantic_label)==self.label_weed] = 2
        semantic_target[np.where(semantic_label)==self.label_sugar_beet] = 3
        return semantic_target


    def _make_stem_target(self, stem_label):
        """Construct mask and offset vectors for mixed classification and regression.
        """
        # TODO
        pass


    def __len__(self):
        return len(self.filenames)
