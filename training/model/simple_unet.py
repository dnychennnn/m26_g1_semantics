"""A simple U-Net type model with two heads for testing.

Author: Jan Quakernack

Note: This module contains parts, which were written for other student projects
conducted by the author.
"""

import torch
from torch import nn

from training import CONFIGS_DIR, load_config
from training.model.layers import ConvBlock, ConvSequence


class SimpleUnet(nn.Module):

    @classmethod
    def from_config(cls):
        config = load_config('simple_unet.yaml')
        return SimpleUnet(**config)


    def __init__(self,
                 in_channels,
                 num_filters_encoder,
                 num_filters_semantic_decoder,
                 num_filters_stem_decoder,
                 num_conv_encoder,
                 num_conv_semantic_decoder,
                 num_conv_stem_decoder,
                 dropout_rate):
        super().__init__()

        # make encoder
        self.encoder = Encoder(in_channels=in_channels,
                               num_filters=num_filters_encoder,
                               num_conv=num_conv_encoder,
                               dropout_rate=dropout_rate)

        decoder_in_channels = num_filters_encoder[-1] # output of encoder
        skip_channels = list(reversed(num_filters_encoder[:-1])) # same for both heads

        # make decoders
        self.semantic_decoder = Decoder(in_channels=decoder_in_channels,
                                        skip_channels=skip_channels,
                                        num_filters=num_filters_semantic_decoder,
                                        num_conv=num_conv_semantic_decoder,
                                        dropout_rate=dropout_rate)

        self.stem_decoder = Decoder(in_channels=decoder_in_channels,
                                    skip_channels=skip_channels,
                                    num_filters=num_filters_stem_decoder,
                                    num_conv=num_conv_stem_decoder,
                                    dropout_rate=dropout_rate)

        # final convolution of both heads to get right number of output dimensions
        self.final_conv_semantic = ConvBlock(in_channels=num_filters_semantic_decoder[-1],
                                             out_channels=3, # three classes: background, weed, sugar beet
                                             kernel_size=1,
                                             padding=0,
                                             activation=None,
                                             dropout_rate=None)

        self.final_conv_stem = ConvBlock(in_channels=num_filters_stem_decoder[-1],
                                         out_channels=3, # three outputs: stem confidence, x offset, y offset
                                         kernel_size=1,
                                         padding=0,
                                         activation=None,
                                         dropout_rate=None)

    def forward(self, x):
        x, skips = self.encoder(x)

        semantic_output = self.semantic_decoder(x, skips)
        semantic_output = self.final_conv_semantic(semantic_output)

        stem_output = self.stem_decoder(x, skips)
        stem_output = self.final_conv_semantic(stem_output)

        stem_keypoint_output = stem_output[:, 0, ...]
        stem_offset_output = stem_output[:, 1:, ...]

        return semantic_output, stem_keypoint_output, stem_offset_output


class Encoder(nn.Module):

    def __init__(self,
                 in_channels,
                 num_filters,
                 num_conv,
                 dropout_rate):
        super(Encoder, self).__init__()

        self.initial_conv = ConvBlock(in_channels=in_channels,
                                      out_channels=num_filters[0],
                                      kernel_size=1,
                                      padding=0,
                                      activation='leaky_relu',
                                      dropout_rate=dropout_rate)

        assert len(num_filters)==len(num_conv)
        num_sequences = len(num_filters)

        sequences = []
        sequence_in_channels = num_filters[0]
        for sequence_index in range(num_sequences):
            sequences.append(ConvSequence(in_channels=sequence_in_channels,
                                          out_channels=num_filters[sequence_index],
                                          num_conv_blocks=num_conv[sequence_index],
                                          activation='leaky_relu',
                                          dropout_rate=dropout_rate))
            sequence_in_channels = num_filters[sequence_index]
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
                 in_channels,
                 skip_channels,
                 num_filters,
                 num_conv,
                 dropout_rate):
      super(Decoder, self).__init__()

      assert len(num_filters)==len(num_conv) and len(num_filters)==len(skip_channels)
      num_sequences = len(num_filters)

      sequences = []
      sequence_in_channels = in_channels
      for sequence_index in range(num_sequences):
          sequences.append(ConvSequence(in_channels=sequence_in_channels+skip_channels[sequence_index],
                                        out_channels=num_filters[sequence_index],
                                        num_conv_blocks=num_conv[sequence_index],
                                        activation='leaky_relu',
                                        dropout_rate=dropout_rate))
          sequence_in_channels = num_filters[sequence_index]

      self.sequences = nn.ModuleList(sequences)


    def forward(self, x, skips):
        for sequence_index, sequence in enumerate(self.sequences):
            x = nn.functional.interpolate(x, size=skips[sequence_index].shape[-2:], mode='nearest')
            x = torch.cat([x, skips[sequence_index]], dim=1)
            x = sequence(x)

        return x
