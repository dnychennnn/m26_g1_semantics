"""A simple U-Net type model with two heads for testing.

Author: Jan Quakernack

Note: This module contains parts, which were written for other student projects
conducted by the author.
"""

import torch
from torch import nn
import numpy as np

from training import CONFIGS_DIR, load_config
from training.models.layers import ConvBlock, ConvSequence
from training.models.postprocessing_modules import StemVoting


class SimpleUnet(nn.Module):

    @classmethod
    def from_config(cls, phase):
        """
        Args:
            phase (str): 'training' or 'deployment'
        """
        config = load_config('training.yaml') if phase=='training' else load_config('deployment.yaml')
        model_config = load_config('simple_unet.yaml')

        model_parameters = {**model_config}

        model_parameters['input_channels'] = config['input_channels']
        model_parameters['input_height'] = config['input_height']
        model_parameters['input_width'] = config['input_width']
        model_parameters['keypoint_radius'] = config['keypoint_radius']

        return SimpleUnet(**model_parameters)


    def __init__(self,
                 input_channels,
                 input_height,
                 input_width,
                 keypoint_radius,
                 num_filters_encoder,
                 num_filters_semantic_decoder,
                 num_filters_stem_decoder,
                 num_conv_encoder,
                 num_conv_semantic_decoder,
                 num_conv_stem_decoder,
                 dropout_rate):
        super().__init__()

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

        # inference module to get the stem position from keypoint confindences and offsets
        self.stem_voting_module = StemVoting(input_width, input_height, keypoint_radius)


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

        if not self.training:
          # we are in evaluation mode
          stem_voting_output = self.stem_voting_module(stem_keypoint_output, stem_offset_output)

          return semantic_output, stem_keypoint_output, stem_offset_output, stem_voting_output

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
            sequences.append(ConvSequence(input_channels=sequence_input_channels,
                                          output_channels=num_filters[sequence_index],
                                          num_conv_blocks=num_conv[sequence_index],
                                          activation='leaky_relu',
                                          dropout_rate=dropout_rate))
            sequence_input_channels = num_filters[sequence_index]
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
