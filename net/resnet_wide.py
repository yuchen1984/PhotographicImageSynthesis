import tensorflow as tf
import numpy as np
from math import ceil

from tflearn.layers.core import dropout, activation
from tflearn.layers.conv import conv_2d, avg_pool_2d, global_avg_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization

def wide_residual_block(incoming, nb_blocks, out_channels, downsample=False,
                        downsample_strides=2, activ='relu', batch_norm=True,
                        apply_dropout=False, keep_prob=0.5,
                        bias=True, weights_init='variance_scaling',
                        bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                        trainable=True, restore=True, reuse=False, scope=None,
                        name="WideResidualBlock"):
  """ Wide Residual Block 

  A wide residual block as described in Wide Residual Networks,
  which is generalised from the residual block as described in
  MSRA's Deep Residual Network paper.
  Full pre-activation architecture is used here.

  Input:
      4-D Tensor [batch, height, width, in_channels].

  Output:
      4-D Tensor [batch, new height, new width, nb_filter].

  Arguments:
      incoming: `Tensor`. Incoming 4-D Layer.
      nb_blocks: `int`. Number of layer blocks.
      out_channels: `int`. The number of convolutional filters of the
          convolution layers.
      downsample: `bool`. If True, apply downsampling using
          'downsample_strides' for strides.
      downsample_strides: `int`. The strides to use when downsampling.
      activation: `str` (name) or `function` (returning a `Tensor`).
          Activation applied to this layer (see tflearn.activations).
          Default: 'linear'.
      batch_norm: `bool`. If True, apply batch normalization.
      bias: `bool`. If True, a bias is used.
      weights_init: `str` (name) or `Tensor`. Weights initialization.
          (see tflearn.initializations) Default: 'uniform_scaling'.
      bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
          (see tflearn.initializations) Default: 'zeros'.
      regularizer: `str` (name) or `Tensor`. Add a regularizer to this
          layer weights (see tflearn.regularizers). Default: None.
      weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
      trainable: `bool`. If True, weights will be trainable.
      restore: `bool`. If True, this layer weights will be restored when
          loading a model.
      reuse: `bool`. If True and 'scope' is provided, this layer variables
          will be reused (shared).
      scope: `str`. Define this layer scope (optional). A scope can be
          used to share variables between layers. Note that scope will
          override name.
      name: A name for this layer (optional). Default: 'ShallowBottleneck'.

  References:
      - Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
          Zhang, Shaoqing Ren, Jian Sun. 2015.
      - Wide Residual Networks, S. Zagoruyko et al. BMVC 2016

  """
  resnet = incoming
  in_channels = incoming.get_shape().as_list()[-1]

  with tf.variable_scope(scope, name, values=[incoming], reuse=reuse) as scope:
    name = scope.name #TODO

    for i in range(nb_blocks):
      identity = resnet

      if not downsample:
          downsample_strides = 1

      if batch_norm:
        resnet = batch_normalization(resnet)
      resnet = activation(resnet, activ)

      resnet = conv_2d(resnet, out_channels, 3,
                       downsample_strides, 'same', 'linear',
                       bias, weights_init, bias_init,
                       regularizer, weight_decay, trainable,
                       restore)
                       
                       
      if apply_dropout:
        resnet = dropout(resnet, keep_prob)

      if batch_norm:
        resnet = batch_normalization(resnet)
      resnet = activation(resnet, activ)

      resnet = conv_2d(resnet, out_channels, 3, 1, 'same',
                       'linear', bias, weights_init,
                       bias_init, regularizer, weight_decay,
                       trainable, restore)

      # Downsampling
      if downsample_strides > 1:
        identity = avg_pool_2d(identity, 1, downsample_strides)

      # Projection to new dimension
      if in_channels != out_channels:
        ch = (out_channels - in_channels)//2
        identity = tf.pad(identity,
                          [[0, 0], [0, 0], [0, 0], [ch, ch]])
        in_channels = out_channels

      resnet = resnet + identity
  return resnet

def create_network(input_data,
                   num_blocks = 2,      # 2 blocks = 16 layers
                   widening_ratio = 8,  # k in the literature
                   dropout = True,
                   keep_probability = 0.7):
  # NB: if k = 1 and dropout = False, this falls back to the original ResNet
  k = widening_ratio
  n = num_blocks  
  network = conv_2d(input_data, 16, 3, regularizer='L2', weight_decay=0.0001)
  network = wide_residual_block(network, n, 16 * k, apply_dropout=dropout, keep_prob=keep_probability)
  network = wide_residual_block(network, 1, 32 * k, downsample=True, apply_dropout=dropout, keep_prob=keep_probability)
  network = wide_residual_block(network, n - 1, 32 * k, apply_dropout=dropout, keep_prob=keep_probability)
  network = wide_residual_block(network, 1, 64 * k, downsample=True, apply_dropout=dropout, keep_prob=keep_probability)
  network = wide_residual_block(network, n - 1, 64 * k, apply_dropout=dropout, keep_prob=keep_probability)
  network = batch_normalization(network)
  network = activation(network, 'relu')
  network = global_avg_pool(network)
  return network
