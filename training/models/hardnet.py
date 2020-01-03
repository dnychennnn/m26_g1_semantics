"""Encoder backbone using harmonic densely connected blocks proposed to reduce memory traffic.

Reference: https://github.com/PingoLH/Pytorch-HarDNet

Code is very similar to the reference implementation on github,
naming was adopted for consistency and to clarify things for ourselves.
"""

import torch
from torch import nn
import numpy as np
import math

from training import CONFIGS_DIR, load_config


class HarDNet(nn.Module):

    @classmethod
    def from_config(cls, architecture_name, phase):
        """
        Args:
            phase (str): 'training' or 'deployment'
        """
        hardnet_config = load_config(architecture_name+'.yaml')

        config = load_config('training.yaml') if phase=='training' else load_config('deployment.yaml')

        hardnet_config.update(config)

        hardnet_parameters = {**hardnet_config}
        return HarDNet(**hardnet_parameters)


    def __init__(self,
                 initial_conv_output_channels,
                 num_conv_blocks_per_stage,
                 output_channels_per_stage,
                 growth_rate_per_stage,
                 do_downsampling_per_stage,
                 provide_as_output_per_stage,
                 channels_weighting_factor,
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
            # print('hardnet stage in, out', block_input_channels, output_channels)

            hard_block = HarDBlock(input_channels=block_input_channels,
                                   num_conv_blocks=num_conv_blocks,
                                   growth_rate=growth_rate,
                                   channels_weighting_factor=channels_weighting_factor,
                                   keep_base=False,
                                   activation='leaky_relu',
                                   dropout_rate=dropout_rate)

            module_list.append(hard_block)
            self.is_output_module.append(False)

            # debug output
            # print('hardblock out', hard_block.get_output_channels())

            # transitional 1x1 convolution to get desired number of output channels

            transition_block = HarDConvBlock(input_channels=hard_block.get_output_channels(),
                                             output_channels=output_channels,
                                             kernel_size=1,
                                             padding=0,
                                             activation=None, # no activation here, different from the reference
                                             dropout_rate=None)

            module_list.append(transition_block)
            self.is_output_module.append(True if provide_as_output else False)

            # downsampling

            if do_downsampling:
                downsampling = HarDDepthWiseConvBlock(input_channels=output_channels,
                                                      output_channels=output_channels,
                                                      stride=2)
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
        # print('hardnet all out:')
        # for output in outputs:
            # print('    * '+str(output.shape))


        return outputs


class HarDBlock(nn.Module):

    def __init__(self,
                 input_channels,
                 num_conv_blocks,
                 growth_rate,
                 channels_weighting_factor,
                 keep_base,
                 activation,
                 dropout_rate):
        super().__init__()

        self.links = []
        conv_blocks = []

        self.output_channels = 0
        self.keep_base = keep_base

        for conv_block_index in range(num_conv_blocks):
            (block_output_channels,
             block_input_channels,
             link,) = self.get_link(conv_block_index=conv_block_index+1,
                                    block_base_channels=input_channels,
                                    growth_rate=growth_rate,
                                    channels_weighting_factor=channels_weighting_factor)
            self.links.append(link)

            # debug output
            # print('block index', conv_block_index)
            # print('block in', block_input_channels)
            # print('block out', block_output_channels)
            # print('block link', link)

            conv_blocks.append(HarDCombinedConvBlock(input_channels=block_input_channels,
                                                     output_channels=block_output_channels,
                                                     activation=activation,
                                                     dropout_rate=dropout_rate))

            if conv_block_index==num_conv_blocks-1 or conv_block_index%2==0:
                # use output of every second block plus output of the final block
                self.output_channels += block_output_channels

        self.conv_blocks = nn.ModuleList(conv_blocks)

        # debug output
        # print('block final out', self.output_channels)


    def get_output_channels(self):
        return self.output_channels


    def get_link(self,
                 conv_block_index,
                 block_base_channels,
                 growth_rate,
                 channels_weighting_factor):
        """
        Returns:
            A tuple of number of output channels, number of input channels
            and list containing the indices of the linked blocks.
        """
        if conv_block_index == 0:
            return (block_base_channels, 0, [],)

        block_output_channels = growth_rate
        link = []

        for wave_index in range(10):
            wave_length = 2**wave_index

            if conv_block_index%wave_length==0:
                linked_block_index = conv_block_index-wave_length
                link.append(linked_block_index)

                if wave_index!=0:
                    block_output_channels *= channels_weighting_factor

        # bring to a close and even integer
        block_output_channels = int(int(block_output_channels+1.0)/2.0)*2

        block_input_channels = 0

        # sum over all linked blocks to get the total number of input channels
        for linked_block_index in link:
            linked_output_channels, _, _ = self.get_link(linked_block_index,
                                                         block_base_channels,
                                                         growth_rate,
                                                         channels_weighting_factor)
            block_input_channels += linked_output_channels

        return (block_output_channels, block_input_channels, link,)


    def forward(self, x):
        block_outputs = [x]

        for conv_block_index, conv_block in enumerate(self.conv_blocks):
            link = self.links[conv_block_index]

            block_inputs = []

            # add output from linked blocks as inputs
            for linked_block_index in link:
                block_inputs.append(block_outputs[linked_block_index])

            if len(block_inputs)>1:
                x = torch.cat(block_inputs, dim=1)
            else:
                x = block_inputs[0]

            # debug output
            # print('hard block link', link)
            # print('hard block intermediate in', x.shape)

            x = conv_block(x)
            block_outputs.append(x)

            # debug output
            # print('hard block intermediate out', x.shape)

            # print('hard block all outputs so far:')
            # for output in block_outputs:
                # print('    * ' + str(output.shape))


        final_outputs = []
        num_block_outputs = len(block_outputs)

        for block_output_index in range(num_block_outputs):
            if ((block_output_index==0 and self.keep_base)
                or block_output_index==num_block_outputs-1
                or block_output_index%2==1):
                final_outputs.append(block_outputs[block_output_index])

        # debug output
        # print('all hard block final outputs:')
        # for output in final_outputs:
            # print('    * ' + str(output.shape))

        final_outputs = torch.cat(final_outputs, dim=1)

        # debug output
        # print('final hard block out', final_outputs.shape)

        return final_outputs


class HarDCombinedConvBlock(nn.Sequential):
    """
    A 1x1 concolutional block and a depth-wise 3x3 convolutional block combined.
    """
    def __init__(self,
                 input_channels,
                 output_channels,
                 activation,
                 dropout_rate):
        super().__init__()

        self.add_module('conv_block', HarDConvBlock(input_channels=input_channels,
                                                    output_channels=output_channels,
                                                    kernel_size=1,
                                                    padding=0,
                                                    activation=activation,
                                                    dropout_rate=dropout_rate))

        self.add_module('depth_wise_conv_block',
                        HarDDepthWiseConvBlock(input_channels=output_channels,
                                               output_channels=output_channels,
                                               stride=1))


class HarDConvBlock(nn.Sequential):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 padding,
                 activation,
                 dropout_rate):
        super().__init__()
        self.add_module('conv', nn.Conv2d(input_channels,
                                          out_channels=output_channels,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          padding=padding,
                                          bias=False))
        self.add_module('batch_norm', nn.BatchNorm2d(output_channels))

        if activation=='relu':
            self.add_module('relu', nn.ReLU())
        elif activation=='leaky_relu':
            self.add_module('leaky_relu', nn.LeakyReLU())
        elif activation=='sigmoid':
            self.add_module('sigmoid', nn.Sigmoid())
        elif activation=='tanh':
            self.add_module('tanh', nn.Tanh())
        elif activation=='softmax':
            self.add_module('softmax', nn.Softmax(dim=1))
        elif activation is None or activation=='none':
            pass
        else:
            warnings.warn("Convolutional block with not-supported activation '{}'.".format(activation))

        if dropout_rate is not None and dropout_rate>0.0:
            self.add_module('dropout', nn.Dropout2d(dropout_rate))


class HarDDepthWiseConvBlock(nn.Sequential):
    def __init__(self,
                 input_channels,
                 output_channels,
                 stride):
        super().__init__()

        self.add_module('depth_wise_conv',
                        nn.Conv2d(input_channels,
                                  out_channels=output_channels,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=1,
                                  groups=output_channels,
                                  bias=False))

        self.add_module('batch_norm',
                        nn.BatchNorm2d(output_channels))

        # note there is no activation applied here

