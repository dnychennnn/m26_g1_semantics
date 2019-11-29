"""Interface to datasets.

Authors: Yung-Yu Chen, Jan Quakernack
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import yaml
from pathlib import Path

from training import load_config, DATA_DIR


class SugarBeetDataset(Dataset):
    """
    Usage:
        dataset = SugarBeetDataset.from_config()
    """

    @classmethod
    def from_config(cls):
        training_config = load_config('training.yaml')
        dataset_config = load_config('sugar_beet_dataset.yaml')

        dataset_parameters = {**dataset_config}
        dataset_parameters['input_height'] = training_config['input_height']
        dataset_parameters['input_width'] = training_config['input_width']
        dataset_parameters['target_height'] = training_config['target_height']
        dataset_parameters['target_width'] = training_config['target_width']
        dataset_parameters['keypoint_radius'] = training_config['keypoint_radius']

        return SugarBeetDataset(**dataset_parameters)


    def __init__(self,
                 label_background, label_ignore, label_weed, label_sugar_beet,
                 input_height, input_width, target_height, target_width,
                 mean_rgb, mean_nir, std_rgb, std_nir,
                 keypoint_radius,
                 path_to_rgb_images, suffix_rgb_images,
                 path_to_nir_images, suffix_nir_images,
                 path_to_semantic_labels, suffix_semantic_labels,
                 path_to_yaml_annotations, suffix_yaml_annotations):
        """Constructor.

        Args:
            mean_xxx, std_xxx: For normalization of intensity values.
            keypoint_radius: Size of stem keypoint given as pixels in target.
            path_to_xxx (str), suffix_xxx (str): Where to find files and suffix
            specific for these files, e.g. 'annotations/dlp/iMapCleaned/',
            '_GroundTruthg_iMap.png'.
        """
        super().__init__()

        self.label_background = label_background
        self.label_ignore = label_ignore
        self.label_weed = label_weed
        self.label_sugar_beet = label_sugar_beet

        self.keypoint_radius = keypoint_radius

        self.path_to_rgb_images = SugarBeetDataset._assure_is_absolute_data_path(Path(path_to_rgb_images))
        self.suffix_rgb_images = suffix_rgb_images

        self.path_to_nir_images = SugarBeetDataset._assure_is_absolute_data_path(Path(path_to_nir_images))
        self.suffix_nir_images = suffix_nir_images

        self.path_to_semantic_labels = SugarBeetDataset._assure_is_absolute_data_path(Path(path_to_semantic_labels))
        self.suffix_semantic_labels = suffix_semantic_labels

        self.path_to_yaml_annotations = SugarBeetDataset._assure_is_absolute_data_path(Path(path_to_yaml_annotations))
        self.suffix_yaml_annotations = suffix_yaml_annotations

        # transformations
        self.input_height = input_height
        self.input_width = input_width
        self.target_height = target_height
        self.target_width = target_width
        self.resize_input = transforms.Resize((input_height, input_width), interpolation=Image.BILINEAR)
        self.resize_target = transforms.Resize((target_height, target_width), interpolation=Image.NEAREST)

        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.mean_nir = mean_nir
        self.std_nir = std_nir

        self.rgb_normalization = transforms.Normalize(mean=mean_rgb, std=std_rgb, inplace=True)
        self.nir_normalization = transforms.Normalize(mean=(mean_nir,), std=(std_nir,), inplace=True)

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
                         and self.get_path_to_yaml_annotations(name).exists())]

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


    def get_path_to_yaml_annotations(self, filename):
        """Construct path to yaml annotations file in this dataset from filename without suffix.
        """
        return self.path_to_yaml_annotations/(filename+self.suffix_yaml_annotations)


    def __getitem__(self, index):
        """Load files for the given index from dataset and convert to tensors.

        Returns:
            Tuple of RGB+NIR input tensor, semantic target.
        """
        filename = self.filenames[index]

        # load images and labels
        rgb_image = Image.open(self.get_path_to_rgb_image(filename)).convert('RGB')
        nir_image = Image.open(self.get_path_to_nir_image(filename)).convert('L')

        original_height, original_width = np.array(rgb_image).shape[:2]
        target_scale_factor_x = self.target_width/original_width
        target_scale_factor_y = self.target_height/original_height

        rgb_image = self.resize_input(rgb_image)
        nir_image = self.resize_input(nir_image)

        # no conversion to RGB here
        semantic_label = Image.open(self.get_path_to_semantic_label(filename))
        semantic_label = self.resize_target(semantic_label)
        semantic_label = np.asarray(semantic_label).astype(np.int)

        # make semantic target
        semantic_target = self._make_semantic_target(semantic_label)

        # make stem target
        with self.get_path_to_yaml_annotations(filename).open('r') as yaml_file:
            annotations_dict = yaml.safe_load(yaml_file)

        stem_positions = SugarBeetDataset._get_stem_positions_from_annotaions_dict(annotations_dict)

        # scale stem positions to target size
        target_stem_positions = [(x*target_scale_factor_x, y*target_scale_factor_y)
                                 for x,y in stem_positions]

        stem_keypoint_target, stem_offset_target = self._make_stem_target(target_stem_positions)

        stem_target = self._make_stem_target(target_stem_positions)

        # random transformations
        # TODO keep in mind that we need to be able to transform stem positions (x, y) in the same way

        # convert input images to tensors
        rgb_tensor = self.pil_to_tensor(rgb_image) # shape (3, input_height, input_width,)
        nir_tensor = self.pil_to_tensor(nir_image) # shape (1, input_height, input_width,)

        # normalization
        rgb_tensor = self.rgb_normalization(rgb_tensor)
        nir_tensor = self.nir_normalization(nir_tensor)

        # concatenate RGB and NIR to single input tensor
        input_tensor = torch.cat([rgb_tensor, nir_tensor], dim=0)

        # convert targets to tensors
        semantic_target_tensor = torch.from_numpy(semantic_target) # shape (target_height, target_width,)

        stem_keypoint_target_tensor = torch.from_numpy(stem_keypoint_target) # shape (1, target_height, target_width,)
        stem_offset_target_tensor = torch.from_numpy(stem_offset_target) # shape (2, target_height, target_width,)

        return input_tensor, semantic_target_tensor, stem_keypoint_target_tensor, stem_offset_target_tensor


    def _make_semantic_target(self, semantic_label):
        """Remap class labels.
        """
        semantic_target = np.zeros_like(semantic_label)
        semantic_target = np.where(semantic_label==self.label_weed, 1, semantic_target)
        semantic_target = np.where(semantic_label==self.label_sugar_beet, 2, semantic_target)
        semantic_target = np.where(semantic_label==self.label_ignore, 3, semantic_target)

        return semantic_target


    @classmethod
    def _get_stem_positions_from_annotaions_dict(cls, annotations_dict):
        """Extract positions from dictionary parsed from annotations file.

        Args:
            annotations_dict (dict): Annotations as parsed from yaml file.

        Returns:
            A list of (x, y) tuples.
        """
        try:
          return [(annotation['stem']['x'], annotation['stem']['y'])
                   for annotation in annotations_dict['annotation']
                   if (annotation['stem']['x']>=0.0
                       and annotation['stem']['y']>=0.0)]
        except KeyError:
          return []


    def _make_stem_target(self, target_stem_positions):
        """Construct keypoint mask and offset vectors for mixed classification and regression.

        Args:
            target_stem_positions (list): A list of (x, y) tuples. These are
            stem positions already transformed to pixel corrdinates in target.

        Returns:
            A tuple of the classification target of shape (target_height, target_width,)
            as an float numpy.array and regression target of shape
            (2, target_height, target_width,) as a float numpy.array.
        """
        keypoint_target = self._make_stem_classification_target(target_stem_positions)
        offset_target = self._make_stem_regression_target(target_stem_positions)

        return keypoint_target, offset_target


    def _make_stem_classification_target(self, target_stem_positions):
        """Construct keypoint mask from stem positions.

        Will put a disk with the keypoint_radius at each stem position.

        Args:
            target_stem_positions (list): A list of (x, y) tuples. These are
            stem positions already transformed to pixel corrdinates in target.

        Returns:
            A keypoint masks of shape (target_height, target_width,) as an
            float32 numpy.array with 0.0=background, 1.0=stem.
        """
        keypoint_target = np.zeros((self.target_height, self.target_width,), dtype=np.float32)

        for stem_x, stem_y in target_stem_positions:
            # put a disk with keypoint_radius at each stem position for classification target
            cv2.circle(keypoint_target,
                       (int(np.round(stem_x)), int(np.round(stem_y))),
                       radius=self.keypoint_radius,
                       color=255, thickness=-1)

        # convert to float and add extra dimension for channels
        keypoint_target = (keypoint_target>0).astype(np.float32)[None, ...]

        # debug output
        # cv2.imshow('keypoint_target', (255*keypoint_target).astype(np.uint8))
        # cv2.waitKey()

        return keypoint_target


    def _make_stem_regression_target(self, target_stem_positions):
        """Construct offset vectors from stem positions.

        Offsets are normalized by keypoint_radius and clipped to be in
        range -1, 1. For pixel, we use the offset for to nearest stem
        only (in case two keypoints overlap).

        Args:
            target_stem_positions (list): A list of (x, y) tuples. These are
            stem positions already transformed to pixel corrdinates in target.

        Returns:
            A float32 numpy.array of shape (2, target_height, target_width,)
            with x and y offsets for each pixel.
        """
        num_stems = len(target_stem_positions)
        if num_stems==0:
            # no stems, return zeros
            return np.zeros((2, self.target_height, self.target_width), dtype=np.float32)

        # x, y for all pixels in target
        xs = np.arange(self.target_width, dtype=np.float32).reshape(1, 1, self.target_width)
        ys = np.arange(self.target_height, dtype=np.float32).reshape(1, self.target_height, 1)

        # x, y, for all stems
        target_stem_positions = np.array(target_stem_positions, dtype=np.float32)
        stem_xs = (target_stem_positions[:, 0]).reshape(num_stems, 1, 1)
        stem_ys = (target_stem_positions[:, 1]).reshape(num_stems, 1, 1)

        # compute offset using broadcasting
        offsets_x = stem_xs-xs
        offsets_y = stem_ys-ys

        # stack, so we have an offset for each pixel
        offsets_x = np.repeat(offsets_x, repeats=self.target_height, axis=1)
        offsets_y = np.repeat(offsets_y, repeats=self.target_width, axis=2)

        # compute norm of offset vector
        offsets_norm = np.sqrt(np.square(offsets_x)+np.square(offsets_y))

        # debug output
        # for stem_index in range(num_stems):
          # target_diagonal = np.linalg.norm([self.target_height, self.target_width])
          # cv2.imshow('offsets_norm', 5.0*offsets_norm[stem_index]/target_diagonal)
          # cv2.imshow('offsets_x', 5.0*offsets_x[stem_index]/target_diagonal)
          # cv2.imshow('offsets_y', 5.0*offsets_y[stem_index]/target_diagonal)
          # cv2.waitKey()

        # get offsets for nearest stem only, the one with minimal norm
        nearest_stem_index = np.argmin(offsets_norm, axis=0)
        # debug_output
        # cv2.imshow('nearest_stem_index', nearest_stem_index/num_stems)
        # cv2.waitKey()
        nearest_stem_index = nearest_stem_index.reshape(1, self.target_height, self.target_width)
        offsets_to_nearest_stem_x = np.take_along_axis(offsets_x, nearest_stem_index, axis=0).reshape(self.target_height, self.target_width)
        offsets_to_nearest_stem_y = np.take_along_axis(offsets_y, nearest_stem_index, axis=0).reshape(self.target_height, self.target_width)

        # debug output
        # target_diagonal = np.linalg.norm([self.target_height, self.target_width])
        # cv2.imshow('offset_to_nearest_stem_x', 5.0*offsets_to_nearest_stem_x/target_diagonal)
        # cv2.imshow('offset_to_nearest_stem_y', 5.0*offsets_to_nearest_stem_y/target_diagonal)
        # cv2.waitKey()

        # normalize offset by keypoint_radius
        offsets_to_nearest_stem_x /= self.keypoint_radius
        offsets_to_nearest_stem_y /= self.keypoint_radius

        # clip to make sure all offsets are in range -1, 1
        # offsets_to_nearest_stem_x = np.clip(offsets_to_nearest_stem_x, -1.0, 1.0)
        # offsets_to_nearest_stem_y = np.clip(offsets_to_nearest_stem_y, -1.0, 1.0)

        # debug output
        # cv2.imshow('offsets_to_nearest_stem_x', 0.5*offsets_to_nearest_stem_x+0.5)
        # cv2.imshow('offsets_to_nearest_stem_y', 0.5*offsets_to_nearest_stem_y+0.5)
        # cv2.waitKey()

        # stack everything to have an array of shape (2, target_height, target_width,)
        offset_target = np.stack([offsets_to_nearest_stem_x, offsets_to_nearest_stem_y], axis=0)

        return offset_target


    def __len__(self):
        return len(self.filenames)


