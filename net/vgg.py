
from tflearn.layers.core import dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge

def create_network(input_data,
                   fine_tuning = False,
                   create_vgg19 = True,
                   keep_prob = 0.5,
                   trailing_fc = 2,
                   trailing_max_pool = True):
  network = conv_2d(input_data, 64, 3, activation='relu', name='conv1_1')
  network = conv_2d(network, 64, 3, activation='relu', name='conv1_2')
  network = max_pool_2d(network, 2, strides=2, name='pool1')

  network = conv_2d(network, 128, 3, activation='relu', name='conv2_1')
  network = conv_2d(network, 128, 3, activation='relu', name='conv2_2')
  network = max_pool_2d(network, 2, strides=2, name='pool2')

  network = conv_2d(network, 256, 3, activation='relu', name='conv3_1')
  network = conv_2d(network, 256, 3, activation='relu', name='conv3_2')
  network = conv_2d(network, 256, 3, activation='relu', name='conv3_3')
  if create_vgg19:
    network = conv_2d(network, 256, 3, activation='relu', name='conv3_4')
  network = max_pool_2d(network, 2, strides=2, name='pool3')

  network = conv_2d(network, 512, 3, activation='relu', name='conv4_1')
  network = conv_2d(network, 512, 3, activation='relu', name='conv4_2')
  network = conv_2d(network, 512, 3, activation='relu', name='conv4_3')
  if create_vgg19:
    network = conv_2d(network, 512, 3, activation='relu', name='conv4_4')
  network = max_pool_2d(network, 2, strides=2, name='pool4')

  network = conv_2d(network, 512, 3, activation='relu', name='conv5_1')
  network = conv_2d(network, 512, 3, activation='relu', name='conv5_2')
  network = conv_2d(network, 512, 3, activation='relu', name='conv5_3')
  if create_vgg19:
    network = conv_2d(network, 512, 3, activation='relu', name='conv5_4')
  if trailing_max_pool:
    network = max_pool_2d(network, 2, strides=2, name='pool5')
  if trailing_fc > 0:
    network = fully_connected(network, 4096, activation='relu',restore=not fine_tuning, name='vgg_fc1')
  if trailing_fc > 1:
    network = dropout(network, keep_prob, name='fc1_dropout')
    network = fully_connected(network, 4096, activation='relu',restore=not fine_tuning, name='vgg_fc2')
  network = dropout(network, keep_prob, name='dropout1')
  return network
