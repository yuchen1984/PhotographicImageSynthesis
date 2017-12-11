#----------------------------------------------------------------------
# *******************************
# size_recommendation.py
# *******************************
# The neural network model used for size classification and regression
# (Author:  Yu Chen)
# *******************************
#----------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from tflearn.layers.core import fully_connected, flatten, dropout
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.metrics import Accuracy, MultiLabelAccuracy
from tflearn.utils import get_incoming_shape

from common.conv_layer_factory import create_cnn_layers
from common.lr_multiplier_factory import create_lr_multipler_dictionary

def create_network(input_data,
                   extra_feature_data,
                   num_classes,
                   network_type = 'googlenet_v',
                   fine_tuning = True,
                   optimizer = 'momentum',
                   loss = 'categorical_crossentropy',
                   keep_prob = 0.5,
                   lr_mult_fc = 10,
                   trailing_max_pool = False,
                   resume = False,
                   fc0_name = 'FullyConnected0',
                   fc1_name = 'FullyConnected1',
                   fc2_name = 'FullyConnected2',
                   fc3_name = 'FullyConnectedOut',
                   fc0_dimension = 0, # 2048
                   fc1_dimension = 4096, # 2048
                   fc2_dimension = 0, # 512
                   lr_filter_string = 'FullyConnected',
                   model_type = 'classification',
                   data_scope = ''):
                   
  if not isinstance(input_data, list):
    input_data = [input_data]
                   
  # Create convolutional layers.                        
  subnetworks = [] 
  with tf.variable_scope(data_scope) as scope:
    subnetwork = create_cnn_layers(network_type, input_data[0], keep_prob)
    subnetwork = flatten(subnetwork)
    subnetworks.append(subnetwork)
  for i in range(len(input_data) - 1):
    with tf.variable_scope(scope, reuse=True):
      subnetwork = create_cnn_layers(network_type, input_data[i + 1], keep_prob)
      subnetwork = flatten(subnetwork)
      subnetworks.append(subnetwork)
      
  if len(subnetworks) > 1:
    network = merge(subnetworks, axis=1, mode='concat')
  else:
    network = subnetworks[0]

  # Feature merge  
  input_shape = get_incoming_shape(network)
  n_inputs = int(np.prod(input_shape[1:]))
  network = tf.reshape(network, [-1, n_inputs])  

  ifrestore = (not fine_tuning) or resume
  if fc0_dimension > 0:
    network = fully_connected(network, fc0_dimension, activation='relu', restore=ifrestore, name=fc0_name)

  print "image feature: ", get_incoming_shape(network)
  print "extra feature: ", get_incoming_shape(extra_feature_data)
  network = merge([network, extra_feature_data], axis=1,mode='concat', name='pool5_7_7_aug')
  
  if fc1_dimension > 0:
    network = fully_connected(network, fc1_dimension, activation='relu', restore=ifrestore, name=fc1_name)
    network = dropout(network, keep_prob, name=('dropout_' + fc1_name))
  if fc2_dimension > 0:
    network = fully_connected(network, fc2_dimension, activation='relu', restore=ifrestore, name=fc2_name)
    
  metric = 'accuracy'
  if model_type == 'classification':
    multi_label = isinstance(num_classes, list)
    if multi_label:
      # To support multiple label classification
      fc_outputs = []
      for i in range(len(num_classes)):
        fc_output = fully_connected(network, num_classes[i], activation='softmax',restore=ifrestore, name=fc3_name + '_' + str(i))
        fc_outputs.append(fc_output)
         
      if len(num_classes) > 1:
        network = merge(fc_outputs, axis=1, mode='concat')
      else:
        network = fc_outputs[0]
      metric = MultiLabelAccuracy(k=len(num_classes), name='m_acc')
    
    else: # Single label classification
      network = fully_connected(network, num_classes, activation='softmax',restore=ifrestore, name=fc3_name)

  elif model_type == 'regression':
    network = fully_connected(network, num_classes, activation='linear',restore=ifrestore, name=fc3_name)
  else:
    raise Exception("Unknown model_type!")
    
  lr_mult = None
  if fine_tuning:
    lr_mult = create_lr_multipler_dictionary([lr_filter_string], lr_mult_fc)
        
  network = regression(network, optimizer=optimizer, loss=loss, metric=metric, restore=ifrestore, lr_multipliers=lr_mult)

  return network
