"""
Reference: https://github.com/PingoLH/Pytorch-HarDNet
"""

import torch
from torch import nn
import numpy as np
import math

from training import CONFIGS_DIR, load_config
from training.models.layers import ConvBlock, ConvSequence


class HarDNet(nn.Module):

    @classmethod
    def from_config(cls, phase):
        """
        Args:
            phase (str): 'training' or 'deployment'
        """
        config = load_config('training.yaml') if phase=='training' else load_config('deployment.yaml')
        model_config = load_config('hardnet.yaml')

        model_parameters = {**model_config}

        model_parameters['phase'] = phase
        model_parameters['input_channels'] = config['input_channels']
        model_parameters['input_height'] = config['input_height']
        model_parameters['input_width'] = config['input_width']

        return HarDNet(**model_parameters)


    def __init__(self,
                 phase,
                 input_channels,
                 input_height,
                 input_width,
                 num_filters_encoder,
                 num_filters_semantic_decoder,
                 num_filters_stem_decoder,
                 num_conv_encoder,
                 num_conv_semantic_decoder,
                 num_conv_stem_decoder,
                 dropout_rate):
        super().__init__()

        self.phase = phase

        # make encoder
        self.encoder = Encoder(input_channels=input_channels,
                               num_filters=num_filters_encoder,
                               num_conv=num_conv_encoder,
                               dropout_rate=dropout_rate)

        decoder_input_channels = num_filters_encoder[-1] # output of encoder
        skip_channels = list(reversed(num_filters_encoder[:-1])) # same for both heads

        # make decoders
        self.semantic_decoder = Decoder(input_channels=decoder_input_channels,
                                        input_height=input_height,
                                        input_width=input_width,
                                        skip_channels=skip_channels,
                                        num_filters=num_filters_semantic_decoder,
                                        num_conv=num_conv_semantic_decoder,
                                        dropout_rate=dropout_rate)

        self.stem_decoder = Decoder(input_channels=decoder_input_channels,
                                    input_height=input_height,
                                    input_width=input_width,
                                    skip_channels=skip_channels,
                                    num_filters=num_filters_stem_decoder,
                                    num_conv=num_conv_stem_decoder,
                                    dropout_rate=dropout_rate)

        # final convolutional sequences
        self.final_sequence_semantic = ConvSequence(input_channels=num_filters_semantic_decoder[-1],
                                                    output_channels=num_filters_semantic_decoder[-1],
                                                    num_conv_blocks=2,
                                                    activation='leaky_relu',
                                                    dropout_rate=None)

        self.final_sequence_stem_keypoint = ConvSequence(input_channels=num_filters_stem_decoder[-1],
                                                         output_channels=num_filters_stem_decoder[-1],
                                                         num_conv_blocks=2,
                                                         activation='leaky_relu',
                                                         dropout_rate=None)

        self.final_sequence_stem_offset = ConvSequence(input_channels=num_filters_stem_decoder[-1],
                                                       output_channels=num_filters_stem_decoder[-1],
                                                       num_conv_blocks=2,
                                                       activation='leaky_relu',
                                                       dropout_rate=None)

        # final convolution with no activation to get right number of output dimensions
        self.final_conv_semantic = ConvBlock(input_channels=num_filters_semantic_decoder[-1],
                                             output_channels=3, # three classes: background, weed, sugar beet
                                             kernel_size=1,
                                             padding=0,
                                             activation=None,
                                             dropout_rate=None)


        self.final_conv_stem_keypoint = ConvBlock(input_channels=num_filters_stem_decoder[-1],
                                                  output_channels=1, # one output: stem confidence
                                                  kernel_size=1,
                                                  padding=0,
                                                  activation=None,
                                                  dropout_rate=None)

        self.final_conv_stem_offset = ConvBlock(input_channels=num_filters_stem_decoder[-1],
                                                output_channels=2, # two outputs: offset x, offset y
                                                kernel_size=1,
                                                padding=0,
                                                activation=None,
                                                dropout_rate=None)

        # softmax of logits of semantic output, applied in evel mode only
        self.softmax_semantic = nn.Softmax(dim=1)

        # sigmoid of logits of stem keypoint output, apllied in eval mode only
        self.sigmoid_stem_keypoint = nn.Sigmoid()


    def forward(self, x):
        x, skips = self.encoder(x)

        semantic_output = self.semantic_decoder(x, skips)
        # print('semantic_output', semantic_output.shape)
        semantic_output = self.final_sequence_semantic(semantic_output)
        # print('semantic_output', semantic_output.shape)
        semantic_output = self.final_conv_semantic(semantic_output)
        # print('semantic_output', semantic_output.shape)

        if not self.training:
          # we are in evaluation mode
          # apply softmax to logits of semantic output
          semantic_output = self.softmax_semantic(semantic_output)

        stem_output = self.stem_decoder(x, skips)
        # print('stem_output', stem_output.shape)

        stem_keypoint_output = self.final_sequence_stem_keypoint(stem_output)
        # print('stem_keypoint_output', stem_keypoint_output.shape)
        stem_keypoint_output = self.final_conv_stem_keypoint(stem_keypoint_output)
        # print('stem_keypoint_output', stem_keypoint_output.shape)

        if not self.training:
          # we are in evaluation mode
          # apply sigmoid to logits of stem keypoint output
          stem_keypoint_output = self.sigmoid_stem_keypoint(stem_keypoint_output)

        stem_offset_output = self.final_sequence_stem_offset(stem_output)
        # print('stem_offset_output', stem_offset_output.shape)
        stem_offset_output = self.final_conv_stem_offset(stem_offset_output)
        # print('stem_offset_output', stem_offset_output.shape)

        return semantic_output, stem_keypoint_output, stem_offset_output


class Encoder(nn.Module):

    def __init__(self,
                 input_channels,
                 num_filters,
                 num_conv,
                 dropout_rate):
        super(Encoder, self).__init__()

        self.initial_conv = ConvBlock(input_channels=input_channels,
                                      output_channels=num_filters[0],
                                      kernel_size=1,
                                      padding=0,
                                      activation='leaky_relu',
                                      dropout_rate=dropout_rate)

        assert len(num_filters)==len(num_conv)
        num_sequences = len(num_filters)

        sequences = []
        sequence_input_channels = num_filters[0]
        for sequence_index in range(num_sequences):
            hard_block = HarDBlock(input_channels=sequence_input_channels,
                                       output_channels=num_filters[sequence_index],
                                       num_conv_blocks=num_conv[sequence_index],
                                       activation='leaky_relu',
                                       dropout_rate=dropout_rate)

            sequences.append(hard_block)

            sequence_input_channels = hard_block.get_output_channels()

            # sequences.append(ConvSequence(input_channels=sequence_input_channels,
                                          # output_channels=num_filters[sequence_index],
                                          # num_conv_blocks=num_conv[sequence_index],
                                          # activation='leaky_relu',
                                          # dropout_rate=dropout_rate))
            # sequence_input_channels = num_filters[sequence_index]
        self.sequences = nn.ModuleList(sequences)


    def forward(self, x):
         x = self.initial_conv(x)
         skips = []
         for sequence_index, sequence in enumerate(self.sequences):
             x = sequence(x)
             if sequence_index!=len(self.sequences)-1:
                 # for the last block do not downsample, do not remember skip
                 skips.insert(0, x) # insert first
                 x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
         return x, skips


class Decoder(nn.Module):

    def __init__(self,
                 input_channels,
                 input_height,
                 input_width,
                 skip_channels,
                 num_filters,
                 num_conv,
                 dropout_rate):
      super(Decoder, self).__init__()

      assert len(num_filters)==len(num_conv) and len(num_filters)==len(skip_channels)
      num_sequences = len(num_filters)

      sequences = []
      upsampling_modules = []
      sequence_input_channels = input_channels
      for sequence_index in range(num_sequences):
          sequences.append(ConvSequence(input_channels=sequence_input_channels+skip_channels[sequence_index],
                                        output_channels=num_filters[sequence_index],
                                        num_conv_blocks=num_conv[sequence_index],
                                        activation='leaky_relu',
                                        dropout_rate=dropout_rate))
          sequence_input_channels = num_filters[sequence_index]

          # infer the downsamples size from input size
          # this is necessary if we use input sizes not dividable by two
          downsampling_count = num_sequences-sequence_index-1
          downsampling_height = input_height
          downsampling_width = input_width
          for index in range(downsampling_count):
            downsampling_height = int(np.floor(0.5*downsampling_height))
            downsampling_width = int(np.floor(0.5*downsampling_width))

          # add upsampling module with the right size
          # we cannot use nn.functional.upsample as this will make the onnx export fail
          upsampling_modules.append(nn.Upsample(size=(downsampling_height, downsampling_width), mode='nearest'))

      self.sequences = nn.ModuleList(sequences)
      self.upsampling_modules = nn.ModuleList(upsampling_modules)


    def forward(self, x, skips):
        for sequence_index, (sequence, upsampling_module) in enumerate(zip(self.sequences, self.upsampling_modules)):
            x = upsampling_module(x)

            x = torch.cat([x, skips[sequence_index]], dim=1)
            x = sequence(x)

        return x


class HarDBlock(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 num_conv_blocks,
                 activation,
                 dropout_rate):
        super().__init__()

        growth_rate = 8
        channels_weighting_factor = 1.8

        self.links = []
        conv_blocks = []

        self.output_channels = 0

        for conv_block_index in range(num_conv_blocks):
            (block_output_channels,
            block_input_channels,
            link,) = self.get_link(conv_block_index=conv_block_index+1,
                                   block_base_channels=input_channels,
                                   growth_rate=growth_rate,
                                   channels_weighting_factor=channels_weighting_factor)
            self.links.append(link)

            # debug output
            print('index', conv_block_index)
            print('in', block_input_channels)
            print('out', block_output_channels)
            print('link', link)
            print('------')

            conv_blocks.append(ConvBlock(input_channels=block_input_channels,
                                         output_channels=block_output_channels,
                                         kernel_size=3,
                                         padding=1,
                                         activation=activation,
                                         dropout_rate=dropout_rate))

            if conv_block_index==num_conv_blocks-1 or conv_block_index%2==0:
                self.output_channels += block_output_channels

        self.conv_blocks = nn.ModuleList(conv_blocks)

        print('final out', self.output_channels)
        print('------')


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
            return block_base_channels, 0, []

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

        return block_output_channels, block_input_channels, link


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

            block_output = conv_block(x)
            block_outputs.append(x)

        final_outputs = []
        num_conv_bocks = len(self.conv_blocks)

        for conv_block_index in num_conv_bocks:
            if conv_block_index==0 or conv_block_index==num_conv_bocks-1 or conv_block_index%2==1:
                final_output.append(block_outputs[conv_block_index])

        return torch.cat(final_outputs, dim=1)

