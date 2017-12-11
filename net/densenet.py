import tensorflow as tf
from tflearn.layers.core import fully_connected
from tflearn.layers.merge_ops import merge

# TODO: Consider reuse corresponding tflearn layers if any
def weight_variable_densenet(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable_densenet(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d_densenet(input, in_features, out_features, kernel_size, with_bias=False):
  W = weight_variable_densenet([ kernel_size, kernel_size, in_features, out_features ])
  conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
  if with_bias:
    return conv + bias_variable_densenet([ out_features ])
  return conv

def batch_activ_conv_densenet(network, in_features, out_features, kernel_size, is_training, keep_prob):
  network = tf.contrib.layers.batch_norm(network, scale=True, is_training=is_training, updates_collections=None)
  network = tf.nn.relu(network)
  network = conv2d_densenet(network, in_features, out_features, kernel_size)
  network = tf.nn.dropout(network, keep_prob)
  return network

def block_densenet(input, layers, in_features, growth, is_training, keep_prob):
  network = input
  features = in_features
  for idx in xrange(layers):
    tmp = batch_activ_conv_densenet(network, features, growth, 3, is_training, keep_prob)
    network = tf.concat(3, (network, tmp))
    features += growth
  return network, features

def avg_pool_densenet(input, s):
  return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

def create_network(input_data,
                   depth = 40,
                   growth_ratio = 12, # k in the literature
                   keep_probability = 0.8): # keep probability in the dropout layers
  layers = int((depth - 4) / 3)
  network = conv2d_densenet(input_data, 3, 16, 3)
  is_training = True
  network, features = block_densenet(network, layers, 16, growth_ratio, is_training, keep_probability)
  network = batch_activ_conv_densenet(network, features, features, 1, is_training, keep_probability)
  network = avg_pool_densenet(network, 2)
  network, features = block_densenet(network, layers, features, growth_ratio, is_training, keep_probability)
  network = batch_activ_conv_densenet(network, features, features, 1, is_training, keep_probability)
  network = avg_pool_densenet(network, 2)
  network, features = block_densenet(network, layers, features, growth_ratio, is_training, keep_probability)

  network = tf.contrib.layers.batch_norm(network, scale=True, is_training=is_training, updates_collections=None)
  network = tf.nn.relu(network)
  network = avg_pool_densenet(network, 8)

  return network
