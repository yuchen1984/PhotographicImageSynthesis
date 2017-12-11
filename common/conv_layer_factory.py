# ***************
# conv_layer_factory.py
# ***************
# The factory for creating convolutional layers of different network architectures. 
# (Author:  Yu Chen)
# ***************

# Import necessary external libraries 
import numpy as np
import tensorflow as tf

import sys
import net.densenet as densenet
import net.googlenet as googlenet
import net.inception_resnet_v2 as inception_resnet_v2
import net.resnet_wide as resnet_wide
import net.vgg as vgg
import net.deeplab_lfov as deeplab_lfov
import net.deeplab_lfov_upsample as deeplab_lfov_upsample
from tflearn.utils import get_incoming_shape

# Create convolutional layers with different network architecture
def create_cnn_layers(network_type, input_data, keep_probability, trailing_max_pool = True,
                      retrain_final_layers = 0, remove_final_layers = 0, num_classes = 2):
                   
  if network_type == 'googlenet_v' or network_type == 'googlenet':
    network = googlenet.create_network(input_data,
                                       keep_probability,
                                       'valid' if network_type == 'googlenet_v' else 'same')
  elif network_type == 'inception_resnet_v2':
    network = inception_resnet_v2.create_network(input_data, keep_probability)
  elif (network_type == 'vgg16' or network_type == 'vgg19'):
    network = vgg.create_network(input_data,
                                 False,
                                 network_type == 'vgg19',
                                 keep_probability,
                                 1 if trailing_max_pool else 0,
                                 trailing_max_pool)
  elif network_type == 'resnet_wide':
    network = resnet_wide.create_network(input_data,
                                         2, # 2 blocks => 16 layers
                                         8, # k = 8 growth rate
                                         True, # has dropout layer in the resnet block
                                         keep_probability)
  elif network_type == 'densenet':
    network = densenet.create_network(input_data,
                                      40,  # L in the litarature, total depth
                                      12,  # k in the literature, growth rate
                                      keep_probability)
  elif network_type == 'deeplab_lfov':
    network = deeplab_lfov.create_network(input_data,
                                          retrain_final_layers,
                                          remove_final_layers,
                                          num_classes,
                                          keep_probability,
                                          3)    # kernel_size = 3
  elif network_type == 'deeplab_lfov_upsample':
    network = deeplab_lfov_upsample.create_network(input_data,
                                          retrain_final_layers,
                                          remove_final_layers,
                                          num_classes,
                                          keep_probability,
                                          3)    # kernel_size = 3
  else:
    raise Exception("Unrecognized network type!")

  print 'Network output dimension: ', get_incoming_shape(network)
  return network
