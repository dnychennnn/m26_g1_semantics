"""Some standard network building blocks.

Author: Jan Quakernack

Note: This module contains parts, which were written for other student projects
conducted by the author.
"""

import torch
from torch import nn
import warnings


class ConvBlock(nn.Sequential):
  def __init__(self,
               input_channels,
               output_channels,
               kernel_size,
               padding,
               activation,
               stride,
               dropout_rate):
      super().__init__()
      self.add_module('conv', nn.Conv2d(input_channels,
                                        out_channels=output_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=True))
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


class ConvSequence(nn.Sequential):
    def __init__(self,
                 input_channels,
                 output_channels,
                 num_conv_blocks,
                 activation,
                 dropout_rate):
      super().__init__()

      for index in range(num_conv_blocks):
          self.add_module('conv_{}'.format(index),
                          ConvBlock(input_channels=input_channels,
                                    output_channels=output_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    dropout_rate=dropout_rate,
                                    activation=activation))
          input_channels=output_channels

