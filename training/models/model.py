"""Common way to get one of our models.

Usage:
    fcn_model = Model.by_name('fcn', phase='training')
    simple_unet_model = Model.by_name('simple_unet', phase='deployment', path_to_weights_file='simple_unet.pth')

Note: Parts adapted from code originally written for MGE-MSR-P-S.
"""

import torch
import numpy as np
from pathlib import Path
import torch
from torch import nn

from training.models.FCN import FCN
from training.models.simple_unet import SimpleUnet
from training import MODELS_DIR, CUDA_DEVICE_NAME, load_config
from training.models.layers import ConvBlock
from training.models.hardnet import HarDNet
from training.models.densenet import DenseNet

class Model(nn.Module):

    @classmethod
    def by_name(cls, architecture_name, phase, verbose=False):
        """Get one of our models by its name.

        Args:
            architecture_name (str): Currently supported 'hardnet29'
            phase (str): 'training' or 'deployment'
            verbose (bool): Print some information.
        """
        if 'hardnet' in architecture_name:
            encoder = HarDNet.from_config(architecture_name, phase)
            model_config = load_config(architecture_name+'.yaml')
            model_parameters = {**model_config}
            model_parameters['encoder'] = encoder
            model = Model(**model_parameters)
        elif 'densenet' in architecture_name:
            encoder = DenseNet.from_config(architecture_name, phase)
            model_config = load_config(architecture_name+'.yaml')
            model_parameters = {**model_config}
            model_parameters['encoder'] = encoder
            model = Model(**model_parameters)
        else:
            raise ValueError("Architechture '{}' is not supported.".format(architecture_name))

        if 'path_to_weights_file' in model_config and model_config['path_to_weights_file']:
            model = Model.load_weights(model,
                                       path_to_weights_file=model_config['path_to_weights_file'],
                                       verbose=verbose)

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


    def __init__(self,
                 input_channels,
                 input_height,
                 input_width,
                 initial_conv_kernel_size,
                 initial_conv_output_channels,
                 initial_conv_stride,
                 output_channels_per_stage,
                 do_downsampling_per_stage,
                 provide_as_output_per_stage,
                 feature_channels,
                 encoder,
                 **extra_arguments):
        super().__init__()

        self.initial_conv_block = ConvBlock(input_channels=input_channels,
                                            output_channels=initial_conv_output_channels,
                                            kernel_size=initial_conv_kernel_size,
                                            padding=initial_conv_kernel_size//2,
                                            stride=initial_conv_stride,
                                            activation='leaky_relu',
                                            dropout_rate=None)

        self.encoder = encoder

        # init upsampling modules and 1x1 convolutions for lateral connections

        output_sizes = self.infer_encoder_output_sizes_(input_height,
                                                        input_width,
                                                        initial_conv_stride,
                                                        do_downsampling_per_stage,
                                                        provide_as_output_per_stage)

        upsampling_modules = []
        for output_size in output_sizes[:-1]:
            upsampling_modules.append(nn.Upsample(size=output_size, mode='nearest'))
        self.upsampling_modules = nn.ModuleList(upsampling_modules[::-1])

        convs_for_lateral_connections = []
        for (output_channels,
            provide_as_output,) in zip(output_channels_per_stage[::-1],
                                       provide_as_output_per_stage[::-1]):

            if provide_as_output:
                convs_for_lateral_connections.append(nn.Conv2d(in_channels=output_channels,
                                                               out_channels=feature_channels,
                                                               kernel_size=1,
                                                               stride=1,
                                                               padding=0,
                                                               bias=True))

        self.convs_for_lateral_connections = nn.ModuleList(convs_for_lateral_connections)

        # network heads

        self.semantic_head = Head(input_channels=feature_channels,
                                  output_channels=3) # three classes: background, weed, sugar_beet

        self.stem_keypoint_head = Head(input_channels=feature_channels,
                                       output_channels=1) # one output: keypoint confidence

        self.stem_offset_head = Head(input_channels=feature_channels,
                                     output_channels=2) # two output: keypoint offsets

        # softmax of logits of semantic output, applied in eval mode only
        self.softmax_semantic = nn.Softmax(dim=1)

        # sigmoid of logits of stem keypoint output, applied in eval mode only
        self.sigmoid_stem_keypoint = nn.Sigmoid()


    def infer_encoder_output_sizes_(self,
                                    input_height,
                                    input_width,
                                    initial_conv_stride,
                                    do_downsampling_per_stage,
                                    provide_as_output_per_stage):
        output_sizes = []

        height = int(np.ceil(input_height/initial_conv_stride).item())
        width = int(np.ceil(input_width/initial_conv_stride).item())

        for (do_downsampling,
             provide_as_output,) in zip(do_downsampling_per_stage,
                                        provide_as_output_per_stage):

            if provide_as_output:
                output_sizes.append((height, width,))

            if do_downsampling:
                height = int(np.ceil(0.5*height).item())
                width = int(np.ceil(0.5*width).item())

        return output_sizes


    def forward(self, x):
        # apply initial convolution
        x = self.initial_conv_block(x)

        # pass through encoder
        encoder_outputs = self.encoder(x)

        # debug output
        # print('all encoder outputs:')
        # for output in encoder_outputs:
            # print('    * '+str(output.shape))

        # pass through decoder

        encoder_outputs = encoder_outputs[::-1] # pyramid top first

        # apply 1x1 convolution to top stage to get number of feature channels right

        x = self.convs_for_lateral_connections[0](encoder_outputs[0])

        for output_index, (encoder_output,
                           upsampling_module,
                           conv,) in enumerate(zip(encoder_outputs[1:],
                                               self.upsampling_modules,
                                               self.convs_for_lateral_connections[1:])):
            # debug output
            # print('x', x.shape)

            x = upsampling_module(x)
            y = conv(encoder_output)

            # debug output
            # print('x', x.shape)
            # print('y', y.shape)

            x += y

        # pass through heads

        semantic_output = self.semantic_head(x)

        # debug output
        # print(semantic_output.shape)

        if not self.training:
          # we are in evaluation mode
          # apply softmax to logits of semantic output
          semantic_output = self.softmax_semantic(semantic_output)

        stem_keypoint_output = self.stem_keypoint_head(x)

        # debug output
        # print(stem_keypoint_output.shape)

        if not self.training:
          # we are in evaluation mode
          # apply sigmoid to logits of stem keypoint output
          stem_keypoint_output = self.sigmoid_stem_keypoint(stem_keypoint_output)

        stem_offset_output = self.stem_offset_head(x)

        # debug output
        # print(stem_offset_output.shape)

        return semantic_output, stem_keypoint_output, stem_offset_output


class Head(nn.Sequential):
    """Network head.

    A 3x3 convolution followed by two 1x1 convolutions.
    """

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.add_module('conv_block_0',
                        ConvBlock(input_channels=input_channels,
                                  output_channels=input_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  dropout_rate=None,
                                  activation='leaky_relu'))

        self.add_module('conv_block_1',
                        ConvBlock(input_channels=input_channels,
                                  output_channels=input_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  dropout_rate=None,
                                  activation='leaky_relu'))

        self.add_module('conv_block_2',
                        ConvBlock(input_channels=input_channels,
                                  output_channels=output_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  dropout_rate=None,
                                  activation=None))
