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

# if we are using pytorch dataloaders, all targets need to have the same size
# so we pad the stem_position_target with zeros to be of shape (MAX_STEM_COUNT, 2)
MAX_STEM_COUNT = 128


class SugarBeetDataset(Dataset):
    """
    Usage:
        dataset = SugarBeetDataset.from_config('train') # or 'val', 'test'

    Note: Parts adapted from code originally written for MGE-MSR-P-S.
    """

    @classmethod
    def from_config(cls, split):
        assert split in ['train', 'val', 'test']

        training_config = load_config('training.yaml')
        dataset_config = load_config('sugar_beet_dataset.yaml')

        dataset_parameters = {**dataset_config}
        dataset_parameters['input_height'] = training_config['input_height']
        dataset_parameters['input_width'] = training_config['input_width']
        dataset_parameters['target_height'] = training_config['target_height']
        dataset_parameters['target_width'] = training_config['target_width']
        dataset_parameters['keypoint_radius'] = training_config['keypoint_radius']

        del dataset_parameters['train']
        del dataset_parameters['val']
        del dataset_parameters['test']

        # only use files of the given split as specified in config
        # we have 100 in 'val', 100 in 'test' and the rest in 'train'
        dataset_parameters['filenames_filter'] = dataset_config[split]

        # data augmentation for train split
        dataset_parameters['random_transformations'] = (RandomTransformations.from_config('train')
                                                        if split=='train' else None)

        return SugarBeetDataset(**dataset_parameters)


    def __init__(self,
                 label_background, label_ignore, label_weed, label_sugar_beet,
                 input_height, input_width, target_height, target_width,
                 mean_rgb, mean_nir, std_rgb, std_nir,
                 keypoint_radius,
                 path_to_rgb_images, suffix_rgb_images,
                 path_to_nir_images, suffix_nir_images,
                 path_to_semantic_labels, suffix_semantic_labels,
                 path_to_yaml_annotations, suffix_yaml_annotations,
                 random_transformations, seed, filenames_filter):
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

        self.random_transformations = random_transformations

        if self.random_transformations is not None:
            self.random = np.random.RandomState(seed)

        self.filenames_filter = filenames_filter

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

        # to provide this for other components
        self.normalization_rgb_dict = {'mean_rgb': self.mean_rgb,
                                       'std_rgb': self.std_rgb}
        self.normalization_nir_dict = {'mean_nir': self.mean_nir,
                                       'std_nir': self.std_nir}

        self.rgb_normalization = transforms.Normalize(mean=mean_rgb, std=std_rgb, inplace=True)
        self.nir_normalization = transforms.Normalize(mean=(mean_nir,), std=(std_nir,), inplace=True)

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

        Also only use those item that are in self.filenames_filter, which we use
        to define our split in train, val and test set.
        """
        # get all filenames of RGB images without suffix
        filenames = [path.name.replace(self.suffix_rgb_images, '')
                     for path in self.path_to_rgb_images.glob('*'+self.suffix_rgb_images)]

        # filter out those for which the corresponding NIR image, semantic label
        # or stem label does not exists
        filenames = [name for name in filenames
                     if (self.get_path_to_nir_image(name).exists()
                         and self.get_path_to_semantic_label(name).exists()
                         and self.get_path_to_yaml_annotations(name).exists()
                         and name in self.filenames_filter)]
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
        rgb_image = cv2.imread(str(self.get_path_to_rgb_image(filename)), cv2.IMREAD_UNCHANGED)[..., ::-1] # BGR to RGB
        nir_image = cv2.imread(str(self.get_path_to_nir_image(filename)), cv2.IMREAD_UNCHANGED)

        original_height, original_width = np.array(rgb_image).shape[:2]
        target_scale_factor_y = self.target_height/original_height
        target_scale_factor_x = self.target_width/original_width

        rgb_image = cv2.resize(rgb_image, (self.input_width, self.input_height), cv2.INTER_LINEAR)
        nir_image = cv2.resize(nir_image, (self.input_width, self.input_height), cv2.INTER_LINEAR)

        semantic_label = cv2.imread(str(self.get_path_to_semantic_label(filename)), cv2.IMREAD_UNCHANGED)
        semantic_label = cv2.resize(semantic_label, (self.input_width, self.input_height), cv2.INTER_NEAREST)

        # make semantic target
        semantic_target = self._make_semantic_target(semantic_label)

        # make stem target
        with self.get_path_to_yaml_annotations(filename).open('r') as yaml_file:
            annotations_dict = yaml.safe_load(yaml_file)

        stem_positions = SugarBeetDataset._get_stem_positions_from_annotaions_dict(annotations_dict)

        # scale stem positions to target size
        stem_position_target = [(x*target_scale_factor_x, y*target_scale_factor_y)
                                 for x, y in stem_positions]
        stem_position_target = np.array(stem_position_target, dtype=np.float32)

        if self.random_transformations is not None:
            # random transformations for data augmentation

            # sample new set of transformation parameters
            self.random_transformations.sample_transformation(self.random.randint(low=0, high=2**32 - 1))

            # transfrom image
            rgb_image = np.array(rgb_image)
            nir_image = np.array(nir_image)

            # pad with zeros
            rgb_image_padded = self._pad_with_zeros(rgb_image)
            nir_image_padded = self._pad_with_zeros(nir_image)
            semantic_target_padded = self._pad_with_zeros(semantic_target)

            rgb_image = self.random_transformations.apply_geometric_transformation_to_image(rgb_image_padded, interpolation=cv2.INTER_LINEAR)
            nir_image = self.random_transformations.apply_geometric_transformation_to_image(nir_image_padded, interpolation=cv2.INTER_LINEAR)
            semantic_target = self.random_transformations.apply_geometric_transformation_to_image(semantic_target_padded, interpolation=cv2.INTER_NEAREST)

            rgb_image = self.random_transformations.apply_color_transformation_to_image(rgb_image)
            nir_image = self.random_transformations.apply_color_transformation_to_image(nir_image)


            if stem_position_target.shape[0]>0:
                # transform stem positions

                # shift positions by padding applied to images
                offset_x = (self.random_transformations.crop_size-self.input_width)//2
                offset_y = (self.random_transformations.crop_size-self.input_height)//2
                stem_position_target += np.array([offset_x, offset_y]).reshape(1, 2)

                # transform positions
                stem_position_target = self.random_transformations.apply_geometric_transformation_to_points(stem_position_target).astype(np.int)

                # only use stems inside transformed image
                inside_image = ((stem_position_target[..., 0]>=0)&(stem_position_target[..., 0]<self.input_width)
                                &(stem_position_target[..., 1]>=0)&(stem_position_target[..., 1]<self.input_height))
                stem_position_target = stem_position_target[inside_image]

        # get keypoint mask and offsets
        stem_keypoint_target, stem_offset_target = self._make_stem_target(stem_position_target)

        # convert input images to tensors
        rgb_tensor = self.pil_to_tensor(rgb_image) # shape (3, input_height, input_width,)
        nir_tensor = self.pil_to_tensor(nir_image) # shape (1, input_height, input_width,)

        # normalization
        rgb_tensor = self.rgb_normalization(rgb_tensor)
        nir_tensor = self.nir_normalization(nir_tensor)

        # concatenate RGB and NIR to single input tensor
        input_tensor = torch.cat([rgb_tensor, nir_tensor], dim=0)

        # convert targets to tensors
        semantic_target_tensor = torch.from_numpy(semantic_target.astype(np.int)) # shape (target_height, target_width,)

        stem_keypoint_target_tensor = torch.from_numpy(stem_keypoint_target.astype(np.float32)) # shape (1, target_height, target_width,)
        stem_offset_target_tensor = torch.from_numpy(stem_offset_target.astype(np.float32)) # shape (2, target_height, target_width,)

        # get the stem count and append zeros to stem_position_target
        # so all loaded tensors have the same shape
        stem_count_target = stem_position_target.shape[0]
        stem_count_target_tensor = torch.tensor(stem_count_target)

        if stem_count_target>0:
            # pad with zeros so we have all targets in a batch of the same shape (MAX_STEM_COUNT, 2,)
            stem_position_target = np.append(stem_position_target.astype(np.float32), np.zeros((MAX_STEM_COUNT-stem_count_target, 2,), dtype=np.float32), axis=0)
        else:
            stem_position_target = np.zeros((MAX_STEM_COUNT, 2,), dtype=np.float32)

        stem_position_target_tensor = torch.from_numpy(stem_position_target)

        target = {'semantic': semantic_target_tensor,
                  'stem_keypoint': stem_keypoint_target_tensor,
                  'stem_offset': stem_offset_target_tensor,
                  'stem_position': stem_position_target_tensor,
                  'stem_count': stem_count_target_tensor}

        return input_tensor, target


    def _make_semantic_target(self, semantic_label):
        """Remap class labels.
        """
        semantic_target = np.zeros_like(semantic_label)
        semantic_target = np.where(semantic_label==self.label_weed, 1, semantic_target)
        semantic_target = np.where(semantic_label==self.label_sugar_beet, 2, semantic_target)
        semantic_target = np.where(semantic_label==self.label_ignore, 3, semantic_target)

        return semantic_target


    def _pad_with_zeros(self, image):
        if len(image.shape)>2:
            image_padded = np.zeros((self.random_transformations.crop_size, self.random_transformations.crop_size, image.shape[2]), dtype=image.dtype)
        else:
            image_padded = np.zeros((self.random_transformations.crop_size, self.random_transformations.crop_size), dtype=image.dtype)

        offset_x = (self.random_transformations.crop_size-self.input_width)//2
        offset_y = (self.random_transformations.crop_size-self.input_height)//2
        image_padded[offset_y:offset_y+self.input_height, offset_x:offset_x+self.input_width] = image

        return image_padded


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


    def _make_stem_target(self, stem_position_target):
        """Construct keypoint mask and offset vectors for mixed classification and regression.

        Args:
            stem_position_target (list or np.array): A list of (x, y) tuples.
            Or a np.array of shape (N, 2), These are stem positions already
            transformed to pixel corrdinates in target.

        Returns:
            A tuple of the classification target of shape (target_height, target_width,)
            as an float numpy.array and regression target of shape
            (2, target_height, target_width,) as a float numpy.array.
        """
        keypoint_target = self._make_stem_classification_target(stem_position_target)
        offset_target = self._make_stem_regression_target(stem_position_target)

        return keypoint_target, offset_target


    def _make_stem_classification_target(self, stem_position_target):
        """Construct keypoint mask from stem positions.

        Will put a disk with the keypoint_radius at each stem position.

        Args:
            stem_position_target (list or np.array): A list of (x, y) tuples.
            Or a np.array of shape (N, 2), These are stem positions already
            transformed to pixel corrdinates in target.

        Returns:
            A keypoint masks of shape (target_height, target_width,) as an
            float32 numpy.array with 0.0=background, 1.0=stem.
        """
        keypoint_target = np.zeros((self.target_height, self.target_width,), dtype=np.float32)

        for stem_x, stem_y in stem_position_target:
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


    def _make_stem_regression_target(self, stem_position_target):
        """Construct offset vectors from stem positions.

        Offsets are normalized by keypoint_radius and clipped to be in
        range -1, 1. For pixel, we use the offset for to nearest stem
        only (in case two keypoints overlap).

        Args:
            stem_position_target (list or np.array): A list of (x, y) tuples.
            Or a np.array of shape (N, 2), These are stem positions already
            transformed to pixel corrdinates in target.

        Returns:
            A float32 numpy.array of shape (2, target_height, target_width,)
            with x and y offsets for each pixel.
        """
        num_stems = len(stem_position_target)
        if num_stems==0:
            # no stems, return zeros
            return np.zeros((2, self.target_height, self.target_width), dtype=np.float32)

        # x, y for all pixels in target
        xs = np.arange(self.target_width, dtype=np.float32).reshape(1, 1, self.target_width)
        ys = np.arange(self.target_height, dtype=np.float32).reshape(1, self.target_height, 1)

        # x, y, for all stems
        stem_position_target = np.array(stem_position_target, dtype=np.float32)
        stem_xs = (stem_position_target[:, 0]).reshape(num_stems, 1, 1)
        stem_ys = (stem_position_target[:, 1]).reshape(num_stems, 1, 1)

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


class RandomTransformations:
    """
    Note: Adapted from code originally written for MGE-MSR-P-S.
    """

    @classmethod
    def from_config(cls, split):
        assert split in ['train'] # no augmentation for val and test split

        config = load_config('training.yaml')

        transformations_config = load_config('random_transformations.yaml')

        parameters = {**transformations_config}

        parameters['input_width'] = config['input_width']
        parameters['input_height'] = config['input_height']

        return RandomTransformations(**parameters)

    def __init__(self,
                 input_width,
                 input_height,
                 rotate,
                 flip_x,
                 flip_y,
                 scale_min,
                 scale_max,
                 shear_min,
                 shear_max,
                 translation_min,
                 translation_max,
                 brightness_min,
                 brightness_max,
                 saturation_min,
                 saturation_max,
                 contrast_min,
                 contrast_max,
                 blur_min,
                 blur_max):

        self.input_width = input_width
        self.input_height = input_height
        self.rotate = rotate
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shear_min = shear_min
        self.shear_max = shear_max
        self.translation_min = translation_min
        self.translation_max = translation_max
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.saturation_min = saturation_min
        self.saturation_max = saturation_max
        self.contrast_min = contrast_min
        self.contrast_max = contrast_max
        self.blur_min = blur_min
        self.blur_max = blur_max

        self.crop_size = self._compute_max_range()

    def _compute_max_range(self):
        '''Pixel size of the maximum square region affetcted by a random transformation.
        '''
        return int(np.ceil(np.reciprocal(self.scale_min)*
                           np.reciprocal(1.0+self.shear_min)*
                           np.maximum(self.input_width, self.input_height)+
                           2.0*np.maximum(np.absolute(self.translation_max),
                                          np.absolute(self.translation_min))))

    def sample_transformation(self, seed):
        '''Sample transformation parameters given the provided seed and set transformation matrix.
        '''
        random = np.random.RandomState(seed)

        # sample transformation matrix for geometric transformation
        scale_x = random.uniform(self.scale_min,
                                 self.scale_max)
        scale_y = random.uniform(self.scale_min,
                                 self.scale_max)
        shear_x = random.uniform(self.shear_min,
                                 self.shear_max)
        shear_y = random.uniform(self.shear_min,
                                 self.shear_max)
        translation_x = random.uniform(self.translation_min,
                                       self.translation_max)
        translation_y = random.uniform(self.translation_min,
                                       self.translation_max)

        rotation_angle = 0.0 # random.uniform(0, 2.0*np.pi) if self.rotate else 1.0
        flip_x = random.choice([1.0, -1.0]) if self.flip_x else 1.0
        flip_y = random.choice([1.0, -1.0]) if self.flip_y else 1.0

        translation_1 = np.eye(3, dtype=np.float)
        translation_1[0, 2] = -0.5*self.crop_size
        translation_1[1, 2] = -0.5*self.crop_size

        scaling = np.eye(3, dtype=np.float)
        scaling[0, 0] = flip_x * scale_x
        scaling[1, 1] = flip_y * scale_y
        scaling[0, 1] = shear_x
        scaling[1, 0] = shear_y
        scaling[2, 2] = 1.0

        rotation = np.eye(3, dtype=np.float)
        rotation[0, 0] = np.cos(rotation_angle)
        rotation[1, 1] = np.cos(rotation_angle)
        rotation[0, 1] = -np.sin(rotation_angle)
        rotation[1, 0] = np.sin(rotation_angle)
        rotation[2, 2] = 1.0

        translation_2 = np.eye(3, dtype=np.float)
        translation_2[0, 2] = 0.5*self.input_width + translation_x
        translation_2[1, 2] = 0.5*self.input_height + translation_y

        self.transformation_matrix = (translation_2@rotation@scaling@translation_1)

        # sample parameters for color transformation and blur
        self.brightness = random.uniform(self.brightness_min,
                                         self.brightness_max)
        self.saturation = random.uniform(self.saturation_min,
                                         self.saturation_max)
        self.contrast = random.uniform(self.contrast_min,
                                       self.contrast_max)
        self.blur = random.uniform(self.blur_min,
                                   self.blur_max)

    def apply_geometric_transformation_to_image(self, image, interpolation):
        image_height, image_width = image.shape[:2]
        assert image_height == self.crop_size and image_width == self.crop_size
        return cv2.warpAffine(image,
                              self.transformation_matrix[:2, :],
                              (self.input_width, self.input_height),
                              flags=interpolation)

    def apply_color_transformation_to_image(self, image):
        # saturation
        if len(image.shape)>2 and image.shape[2]==3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip((hsv[..., 1]+255.0*self.saturation), 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # contrast
        mean = (np.mean(image) if len(image.shape)==2
                else np.mean(image.reshape(self.input_height*self.input_width, image.shape[2]),
                             axis=0).reshape(1, 1, image.shape[2]))
        image = np.clip((1.0+self.contrast)*(image-mean)+mean, 0, 255).astype(np.uint8)

        # brightness
        image = np.clip(image + 255.0*self.brightness, 0, 255).astype(np.uint8)

        # blur
        if self.blur > 0.0:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=self.blur)

        return image

    def apply_geometric_transformation_to_points(self, points):
        num_points = points.shape[0]

        # using homogeneous coordinates
        points = np.stack([points[:, 0], points[:, 1], np.ones((num_points, ), dtype=np.float)], axis=-1)

        transformed_points = ((self.transformation_matrix@points.T).T)[:, :2]
        return transformed_points
