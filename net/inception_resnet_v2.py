import tensorflow as tf

import tflearn
import tflearn.activations as activations

# Data loading and preprocessing
from tflearn.activations import relu
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import avg_pool_2d, conv_2d, max_pool_2d
from tflearn.layers.core import dropout, flatten, fully_connected, input_data
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization
from tflearn.utils import repeat

def block35(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None, name='Conv2d_1x1')))
    tower_conv1_0 = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None,name='Conv2d_0a_1x1')))
    tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 32, 3, bias=False, activation=None,name='Conv2d_0b_3x3')))
    tower_conv2_0 = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
    tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2_0, 48,3, bias=False, activation=None, name='Conv2d_0b_3x3')))
    tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 64,3, bias=False, activation=None, name='Conv2d_0c_3x3')))
    tower_mixed = merge([tower_conv, tower_conv1_1, tower_conv2_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net

def block17(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_1x1')))
    tower_conv_1_0 = relu(batch_normalization(conv_2d(net, 128, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
    tower_conv_1_1 = relu(batch_normalization(conv_2d(tower_conv_1_0, 160,[1,7], bias=False, activation=None,name='Conv2d_0b_1x7')))
    tower_conv_1_2 = relu(batch_normalization(conv_2d(tower_conv_1_1, 192, [7,1], bias=False, activation=None,name='Conv2d_0c_7x1')))
    tower_mixed = merge([tower_conv,tower_conv_1_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net


def block8(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_1x1')))
    tower_conv1_0 = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
    tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 224, [1,3], bias=False, activation=None, name='Conv2d_0b_1x3')))
    tower_conv1_2 = relu(batch_normalization(conv_2d(tower_conv1_1, 256, [3,1], bias=False, name='Conv2d_0c_3x1')))
    tower_mixed = merge([tower_conv,tower_conv1_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net

def create_network(input_data,
                   keep_prob = 0.5):
  conv1a_3_3 = relu(batch_normalization(conv_2d(input_data, 32, 3, strides=2, bias=False, padding='VALID',activation=None,name='Conv2d_1a_3x3')))
  conv2a_3_3 = relu(batch_normalization(conv_2d(conv1a_3_3, 32, 3, bias=False, padding='VALID',activation=None, name='Conv2d_2a_3x3')))
  conv2b_3_3 = relu(batch_normalization(conv_2d(conv2a_3_3, 64, 3, bias=False, activation=None, name='Conv2d_2b_3x3')))
  maxpool3a_3_3 = max_pool_2d(conv2b_3_3, 3, strides=2, padding='VALID', name='MaxPool_3a_3x3')
  conv3b_1_1 = relu(batch_normalization(conv_2d(maxpool3a_3_3, 80, 1, bias=False, padding='VALID',activation=None, name='Conv2d_3b_1x1')))
  conv4a_3_3 = relu(batch_normalization(conv_2d(conv3b_1_1, 192, 3, bias=False, padding='VALID',activation=None, name='Conv2d_4a_3x3')))
  maxpool5a_3_3 = max_pool_2d(conv4a_3_3, 3, strides=2, padding='VALID', name='MaxPool_5a_3x3')

  tower_conv = relu(batch_normalization(conv_2d(maxpool5a_3_3, 96, 1, bias=False, activation=None, name='Conv2d_5b_b0_1x1')))

  tower_conv1_0 = relu(batch_normalization(conv_2d(maxpool5a_3_3, 48, 1, bias=False, activation=None, name='Conv2d_5b_b1_0a_1x1')))
  tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 64, 5, bias=False, activation=None, name='Conv2d_5b_b1_0b_5x5')))

  tower_conv2_0 = relu(batch_normalization(conv_2d(maxpool5a_3_3, 64, 1, bias=False, activation=None, name='Conv2d_5b_b2_0a_1x1')))
  tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2_0, 96, 3, bias=False, activation=None, name='Conv2d_5b_b2_0b_3x3')))
  tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 96, 3, bias=False, activation=None,name='Conv2d_5b_b2_0c_3x3')))

  tower_pool3_0 = avg_pool_2d(maxpool5a_3_3, 3, strides=1, padding='same', name='AvgPool_5b_b3_0a_3x3')
  tower_conv3_1 = relu(batch_normalization(conv_2d(tower_pool3_0, 64, 1, bias=False, activation=None,name='Conv2d_5b_b3_0b_1x1')))

  tower_5b_out = merge([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1], mode='concat', axis=3)

  net = repeat(tower_5b_out, 10, block35, scale=0.17)

  tower_conv = relu(batch_normalization(conv_2d(net, 384, 3, bias=False, strides=2,activation=None, padding='VALID', name='Conv2d_6a_b0_0a_3x3')))
  tower_conv1_0 = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, activation=None, name='Conv2d_6a_b1_0a_1x1')))
  tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 256, 3, bias=False, activation=None, name='Conv2d_6a_b1_0b_3x3')))
  tower_conv1_2 = relu(batch_normalization(conv_2d(tower_conv1_1, 384, 3, bias=False, strides=2, padding='VALID', activation=None,name='Conv2d_6a_b1_0c_3x3')))
  tower_pool = max_pool_2d(net, 3, strides=2, padding='VALID',name='MaxPool_1a_3x3')
  net = merge([tower_conv, tower_conv1_2, tower_pool], mode='concat', axis=3)
  net = repeat(net, 20, block17, scale=0.1)

  tower_conv = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
  tower_conv0_1 = relu(batch_normalization(conv_2d(tower_conv, 384, 3, bias=False, strides=2, padding='VALID', activation=None,name='Conv2d_0a_1x1')))

  tower_conv1 = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, padding='VALID', activation=None,name='Conv2d_0a_1x1')))
  tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1,288,3, bias=False, strides=2, padding='VALID',activation=None, name='COnv2d_1a_3x3')))

  tower_conv2 = relu(batch_normalization(conv_2d(net, 256,1, bias=False, activation=None,name='Conv2d_0a_1x1')))
  tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2, 288,3, bias=False, name='Conv2d_0b_3x3',activation=None)))
  tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 320, 3, bias=False, strides=2, padding='VALID',activation=None, name='Conv2d_1a_3x3')))

  tower_pool = max_pool_2d(net, 3, strides=2, padding='VALID', name='MaxPool_1a_3x3')
  net = merge([tower_conv0_1, tower_conv1_1,tower_conv2_2, tower_pool], mode='concat', axis=3)

  net = repeat(net, 9, block8, scale=0.2)
  net = block8(net, activation=None)

  net = relu(batch_normalization(conv_2d(net, 1536, 1, bias=False, activation=None, name='Conv2d_7b_1x1')))
  net = avg_pool_2d(net, net.get_shape().as_list()[1:3],strides=2, padding='VALID', name='AvgPool_1a_8x8')
  net = flatten(net)
  net = dropout(net, keep_prob)
  
  return net
