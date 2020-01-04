"""DenseNet as encoder backbone.

Reference: https://arxiv.org/pdf/1608.06993.pdf
"""

import torch
from torch import nn
import numpy as np
import math

from training import CONFIGS_DIR, load_config


class DenseNet(nn.Module):

    @classmethod
    def from_config(cls, architecture_name, phase):
        """
        Args:
            phase (str): 'training' or 'deployment'
        """
        densenet_config = load_config(architecture_name+'.yaml')

        config = load_config('training.yaml') if phase=='training' else load_config('deployment.yaml')

        densenet_config.update(config)

        densenet_parameters = {**densenet_config}
        return DenseNet(**densenet_parameters)


    def __init__(self,
                 initial_conv_output_channels,
                 num_conv_blocks_per_stage,
                 output_channels_per_stage,
                 growth_rate_per_stage,
                 do_downsampling_per_stage,
                 provide_as_output_per_stage,
                 dropout_rate,
                 **extra_arguments):
        super().__init__()

        module_list = []
        block_input_channels = initial_conv_output_channels

        self.is_output_module = []

        for (num_conv_blocks,
             output_channels,
             growth_rate,
             do_downsampling,
             provide_as_output,) in zip(num_conv_blocks_per_stage,
                                        output_channels_per_stage,
                                        growth_rate_per_stage,
                                        do_downsampling_per_stage,
                                        provide_as_output_per_stage):

            # debug output
            # print('densenet stage in, out', block_input_channels, output_channels)

            dense_block = DenseBlock(input_channels=block_input_channels,
                                     num_conv_blocks=num_conv_blocks,
                                     growth_rate=growth_rate,
                                     activation='leaky_relu',
                                     dropout_rate=dropout_rate)

            module_list.append(dense_block)
            self.is_output_module.append(False)

            # debug output
            # print('denseblock out', dense_block.get_output_channels())

            transition_block = DenseTransitionBlock(input_channels=dense_block.get_output_channels(),
                                                    output_channels=output_channels)



            module_list.append(transition_block)
            self.is_output_module.append(True if provide_as_output else False)

            # downsampling

            if do_downsampling:
                downsampling = DenseDownsamplingBlock(input_channels=output_channels,
                                                      output_channels=output_channels)

                module_list.append(downsampling)
                self.is_output_module.append(False)

            block_input_channels = output_channels

        self.module_list = nn.ModuleList(module_list)


    def forward(self, x):
        outputs = []
        for module, provide_as_ouput in zip(self.module_list, self.is_output_module):
            x = module(x)

            if provide_as_ouput:
                outputs.append(x)

        # debug output
        # print('densenet all out:')
        # for output in outputs:
            # print('    * '+str(output.shape))


        return outputs


class DenseBlock(nn.Module):

    def __init__(self,
                 input_channels,
                 num_conv_blocks,
                 growth_rate,
                 activation,
                 dropout_rate):
        super().__init__()

        conv_blocks = []

        for conv_block_index in range(num_conv_blocks):
            conv_block = DenseConvBlockWithBottleneck(input_channels+conv_block_index*growth_rate,
                                                      growth_rate=growth_rate,
                                                      activation=activation,
                                                      dropout_rate=dropout_rate)
            conv_blocks.append(conv_block)

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.output_channels = input_channels+num_conv_blocks*growth_rate


    def get_output_channels(self):
        return self.output_channels


    def forward(self, x):
        block_outputs = [x]

        for conv_block_index, conv_block in enumerate(self.conv_blocks):
            x = torch.cat(block_outputs, dim=1)
            block_outputs.append(conv_block(x))

        # print('dense block all outputs so far:')
        # for output in block_outputs:
            # print('    * '+str(output.shape))

        final_output = torch.cat(block_outputs, dim=1)

        # print('final dense block out', final_output.shape)

        return final_output


class DenseConvBlockWithBottleneck(nn.Sequential):
    """
    A 1x1 and a 3x3 convolution combined. Denoted as DenseNet-B in the reference.
    """
    def __init__(self,
                 input_channels,
                 growth_rate,
                 activation,
                 dropout_rate):
        super().__init__()

        self.add_module('batch_norm_1', nn.BatchNorm2d(input_channels))

        if activation=='relu':
            self.add_module('relu_1', nn.ReLU())
        elif activation=='leaky_relu':
            self.add_module('leaky_relu_1', nn.LeakyReLU())
        elif activation is None or activation=='none':
            pass
        else:
            warnings.warn("Convolutional block with not-supported activation '{}'.".format(activation))

        self.add_module('conv_1', nn.Conv2d(input_channels,
                                            out_channels=4*growth_rate,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=False))

        self.add_module('batch_norm_2', nn.BatchNorm2d(4*growth_rate))

        if activation=='relu':
            self.add_module('relu_2', nn.ReLU())
        elif activation=='leaky_relu':
            self.add_module('leaky_relu_2', nn.LeakyReLU())
        elif activation is None or activation=='none':
            pass
        else:
            warnings.warn("Convolutional block with not-supported activation '{}'.".format(activation))

        self.add_module('conv_2', nn.Conv2d(4*growth_rate,
                                            out_channels=growth_rate,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False))

        if dropout_rate is not None and dropout_rate>0.0:
            self.add_module('dropout', nn.Dropout2d(dropout_rate))


class DenseTransitionBlock(nn.Sequential):
    def __init__(self,
                 input_channels,
                 output_channels):
        super().__init__()

        self.add_module('batch_norm', nn.BatchNorm2d(input_channels))

        self.add_module('conv', nn.Conv2d(input_channels,
                                          out_channels=output_channels,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=False))


class DenseDownsamplingBlock(nn.Sequential):
    """
    Note we use batch norm and a 3x3 convolution with stride 2 here instead
    of average pooling as in the reference.
    """
    def __init__(self,
                 input_channels,
                 output_channels):
        super().__init__()

        self.add_module('batch_norm', nn.BatchNorm2d(input_channels))

        self.add_module('conv', nn.Conv2d(input_channels,
                                          out_channels=output_channels,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          bias=False))

