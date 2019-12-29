"""Common way to get one of our models.

Usage:
    fcn_model = Model.by_name('fcn', phase='training')
    simple_unet_model = Model.by_name('simple_unet', phase='deployment', path_to_weights_file='simple_unet.pth')

Author: Jan Quakernack

Note: Parts adapted from code originally written for MGE-MSR-P-S.
"""

import torch
import numpy as np
from pathlib import Path

from training.models.FCN import FCN
from training.models.simple_unet import SimpleUnet
from training.models.hardnet import HarDNet
from training import MODELS_DIR, CUDA_DEVICE_NAME

class Model:

    @classmethod
    def by_name(cls, architecture_name, phase, path_to_weights_file=None, verbose=False):
        """Get one of our models by its name.

        Args:
            architecture_name (str): Currently supported 'fcn' and 'simple_unet'.
            phase (str): 'training' or 'deployment'
            path_to_weights_file (str or pathlib.Path): Weights as .pth file to load.
            Assumed to be in MODELS_DIR/path_to_weights_file.
            verbose (bool): Print some information.
        """
        if architecture_name=='fcn':
            model = FCN.from_config(phase)
        elif architecture_name=='simple_unet':
            model = SimpleUnet.from_config(phase)
        elif architecture_name=='hardnet':
            model = HarDNet.from_config(phase)
        else:
            raise ValueError("Architechture '{}' is not supported.".format(architecture_name))

        if path_to_weights_file is not None and path_to_weights_file:
            model = Model.load_weights(model, path_to_weights_file=path_to_weights_file, verbose=verbose)

        if verbose:
            trainable_parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
            num_trainable_parameters = sum([np.prod(parameter.size()) for parameter in trainable_parameters])
            print('Number of trainable model parameters: {}'.format(num_trainable_parameters))

        if phase=='training':
          model = model.train()
        elif phase=='deployment':
          model = model.eval()

        return model


    @classmethod
    def load_weights(cls, model, path_to_weights_file, load_parts=True, verbose=False):
        """Load model weights from .pth file.

        If path_to_weights_file is relative, assume weights are in MODELS_DIR/path_to_weights_file.

        Args:
            path_to_weights_file (str or pathlib.Path): Weights as .pth file to load.
            verbose (bool): Print some information.
        """
        path_to_weights_file = Path(path_to_weights_file)

        if not path_to_weights_file.is_absolute():
            path_to_weights_file = MODELS_DIR/path_to_weights_file

        if verbose:
            print('Load weights from {}.'.format(path_to_weights_file))

        device = torch.device(CUDA_DEVICE_NAME if torch.cuda.is_available() else 'cpu')
        model.to(device)

        if load_parts:
            model_dict = model.state_dict()

            # try to load those part of an existing model that match the architecture
            # Reference: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
            pretrained_dict = torch.load(path_to_weights_file, map_location=device)

            no_correspondence = [key for key, value in pretrained_dict.items()
                                 if key not in model_dict or model_dict[key].shape!=value.shape]

            if len(no_correspondence)>0:
                print('Cannot load layers:')
                for key in no_correspondence:
                    print(' * '+key)

            pretrained_dict = {key: value for key, value in pretrained_dict.items()
                               if key in model_dict and model_dict[key].shape==value.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            model_dict = torch.load(path_to_weights_file, map_location=device)
            model.load_state_dict(model_dict)

        return model


