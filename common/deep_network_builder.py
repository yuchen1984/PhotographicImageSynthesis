# ***************
# deep_network_builder.py
# ***************
# The generic builder module to create various deep network architectures for
#  deep learning training and testing. 
# (Author:  Yu Chen)
# ***************

# Import necessary external libraries 
import numpy as np
import tensorflow as tf
from tflearn.layers.core import fully_connected, activation, flatten, input_data, dropout
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.metrics import Accuracy, MultiLabelAccuracy
from tflearn.objectives import categorical_crossentropy
from tflearn.utils import get_incoming_shape

import sys
from common.conv_layer_factory import create_cnn_layers
from common.lr_multiplier_factory import create_lr_multipler_dictionary
from common.optimizer_factory import create_optimizer
from common.triplet_block import create_triplet_block

DEFAULT_SCOPE = ''

# create TensorFlow placeholders for the multiple-parallel data input 
def create_multiple_data_inputs(image_height, image_width, image_depth,
                                count = 1, data_preprocessing = None, data_augmentation = None, data_scope = DEFAULT_SCOPE):
  if count <= 0:
    raise Exception('The argument count must be positive!')
    
  # create the data input placeholders
  data_inputs = [] 
  with tf.variable_scope(data_scope) as scope_in:
    for i in range(count):
      with tf.variable_scope(scope_in, reuse=(i>0)):
        data_input = input_data(shape=[None, image_height, image_width, image_depth],
                                data_preprocessing=data_preprocessing,
                                data_augmentation=data_augmentation)
        data_inputs.append(data_input)
  return data_inputs

# Create fully-connected layers and final regressor after different network architectures
def create_fc_and_regressor_layers(network,
                                   num_classes,
                                   optimizer = 'momentum',
                                   loss = None,
                                   fine_tuning = False,
                                   label_type = 'single_multiclass',
                                   lr_mult_fc = 10,
                                   resume = False,
                                   lr_filter_string = "FullyConnected"):
  if label_type == None:
    return network                                 
                                   
  ifrestore = (not fine_tuning) or resume
  if label_type == 'multi_binary':
    # Multiple binary labels
    network = fully_connected(network, num_classes, activation='sigmoid',restore=ifrestore, name=lr_filter_string)
    metric = MultiLabelAccuracy(k=num_classes, name='m_acc', binary=True)
  elif label_type == 'multi_multiclass':
    # To support multiple multi-class label classification
    fc_outputs = []
    for i in range(len(num_classes)):
      fc_output = fully_connected(network, num_classes[i], activation='softmax',restore=ifrestore, name = (lr_filter_string + str(i)))
      fc_outputs.append(fc_output)
    network = merge(fc_outputs, axis=1, mode='concat')   
    metric = MultiLabelAccuracy(k=len(num_classes), name='m_acc')
  elif label_type == 'single_multiclass': # Single multi-class label classification
    network = fully_connected(network, num_classes, activation='softmax',restore=ifrestore, name=lr_filter_string)
    metric = 'accuracy'
  elif label_type == 'single_binary': # Single binary label classification
    network = fully_connected(network, num_classes, activation='sigmoid',restore=ifrestore, name=lr_filter_string)
    metric = 'accuracy'
  elif label_type == 'hybrid_category_binary':
    # Hybrid attribute and categorical labels
    network_attribute = fully_connected(network, num_classes[0], activation='sigmoid',restore=ifrestore, name= (lr_filter_string + 'Attribute'))
    network_category = fully_connected(network, num_classes[1], activation='softmax',restore=ifrestore, name=(lr_filter_string + 'Category'))
    print 'Feature dimension: ', get_incoming_shape(network_attribute), get_incoming_shape(network_category)
    network = merge([network_attribute, network_category], axis=1, mode='concat')   
    metric = MultiLabelAccuracy(k=(num_classes[0] + num_classes[1]), name='m_acc', binary=True)
  elif label_type == 'regressor_only': # Only build the regression layer
    metric = 'accuracy'
  else:
    raise Exception('Unknown label type')
  
  lr_mult = None
  if fine_tuning:
    lr_mult = create_lr_multipler_dictionary([lr_filter_string], lr_mult_fc)
        
  if loss == None:
    loss = 'categorical_crossentropy'
        
  network = regression(network, optimizer=optimizer, loss=loss, metric=metric, restore=ifrestore, lr_multipliers=lr_mult)
  return network
    
# A factory to create deep NNs of different network architecture with
# different optimization parameters
def create_dnn(network_type,
               input_datas,  # a single or a list of input-data modules
               num_classes,
               optimizer_config = None,
               is_training = True,
               fine_tuning = False,
               lr_mult_fc = 10,
               loss_func = 'categorical_crossentropy',
               resume = False,
               task_type = None,
               label_type = None,
               weight_triplet = 25,
               fc1_name = 'FullyConnected1',
               fc2_name = 'FullyConnected2',
               fc1_dimension = 0, # 2048
               fc2_dimension = 0, # 512
               weight_sharing = 'all',
               data_scope = DEFAULT_SCOPE):
               
  # Create an optimizer from the specified JSON config file
  optimizer = "Momentum" if not optimizer_config else create_optimizer(optimizer_config)
  
  # Specify different network architectures. Normalize to lower cases
  network_type = network_type.lower()
  
  # Set keep probability for drop-out
  keep_probability = 0.6 if is_training else 1.0
  ifrestore = (not fine_tuning) or resume

  with tf.variable_scope(data_scope) as scope:
    if task_type == 'triplet':
      if len(input_datas) < 3:
        raise Exception('For triplet similarity learning, at least 3 parallel data inputs are required')
        
      # create triplet network for similarity metric learning
      network = create_triplet_block(network_type = network_type,
                                    input_data1 = input_datas[0],
                                    input_data2 = input_datas[1],
                                    input_data3 = input_datas[2],
                                    weight_sharing = weight_sharing, 
                                    fc1_dimension = fc1_dimension,
                                    fc2_dimension = fc2_dimension,
                                    fc1_name = fc1_name,
                                    fc2_name = fc2_name,
                                    keep_probability = keep_probability,
                                    ifrestore = ifrestore,
                                    scope_enforced = scope,
                                    reuse_first_cnn = False)
      label_type = 'regressor_only'

    elif task_type == 'hybrid':
      # Multi-task weight-sharing learning: category + attribute + triplet
      if len(input_datas) < 3:
        raise Exception('For multi-task triplet similarity learning and attribute classification, at least 4 parallel data inputs are required')
      
      # Create the weight sharing conv network for triplet similarity learning
      network_triplet = create_triplet_block(network_type = network_type,
                                             input_data1 = input_datas[1],
                                             input_data2 = input_datas[2],
                                             input_data3 = input_datas[3],
                                             weight_sharing = weight_sharing, 
                                             fc1_dimension = fc1_dimension,
                                             fc2_dimension = fc2_dimension,
                                             fc1_name = fc1_name,
                                             fc2_name = fc2_name,
                                             keep_probability = keep_probability,
                                             scope_enforced = scope,
                                             reuse_first_cnn = False)
                                                   
      # Create the weight sharing conv network for attribute and category prediction
      with tf.variable_scope(scope, reuse=True):
        network_ca = create_cnn_layers(network_type, input_datas[0], keep_probability, False)
        network_ca = flatten(network_ca)        
        if weight_sharing == 'all':
          network_ca = fully_connected(network_ca, fc1_dimension, activation='relu',restore=ifrestore, name=fc1_name)
   
      # Create the rest FC layers and regressors attaching to the convolutional layers
      regress_attribute_category = create_fc_and_regressor_layers(network_ca,
                                                                  num_classes,
                                                                  optimizer,
                                                                  loss_func,
                                                                  fine_tuning,
                                                                  label_type,
                                                                  lr_mult_fc,
                                                                  resume,
                                                                  "FCCategory")
                                                                  
      # Create the regressors attaching to the convolutional layers
      loss_triplet = lambda y_pred, y_true: categorical_crossentropy(y_pred, y_true) * weight_triplet
      regress_triplet = create_fc_and_regressor_layers(network_triplet,
                                                       2,
                                                       optimizer,
                                                       loss_triplet,
                                                       fine_tuning,
                                                       'regressor_only',
                                                       lr_mult_fc,
                                                       resume,
                                                       "FullyConnected")
                                          
      network = merge([regress_attribute_category, regress_triplet], axis=1, mode='concat')
      label_type = None
      
    elif task_type == 'feature_extraction' or task_type == 'feature_extraction_multitask':
      # Use the pre-trained attribute classification or similarity metric network for feature extraction
      if is_training:
        raise Exception('feature extraction networks can only be used in the testing stage.')
      if isinstance(input_datas, list):
        input_datas = input_datas[0]

      network = create_cnn_layers(network_type, input_datas, keep_probability)
      network = flatten(network)
      # Add a couple more FC layers for similarity metric embedding
      multitask = (label_type == 'feature_extraction_multitask')
      if fc1_dimension > 0:
        network = fully_connected(network, fc1_dimension, activation=('relu' if (fc2_dimension > 0 or multitask) else 'linear'), restore=ifrestore,
                                  name=(fc1_name if weight_sharing != None else (fc1_name + 's')))
      if fc2_dimension > 0:
        network = fully_connected(network, fc2_dimension, activation=('relu' if multitask else 'linear'), restore=ifrestore,
                                  name=(fc2_name if weight_sharing == 'all' else (fc2_name + 's')))
      if multitask:                           
        network = create_fc_and_regressor_layers(network, num_classes, optimizer, loss_func, fine_tuning, label_type, lr_mult_fc, resume, "FCCategory")
      
      # This will skip the regressor contruction and return the feature vector
      label_type = None
      
    elif task_type == 'classification':
      # Create normal deep network for attribute classification
      if not isinstance(input_datas, list):
        raise Exception('Multiple input flows need to be provided.')
        
      # Regularize the input
      if label_type == 'multi_multiclass' and not isinstance(num_classes, list):
        num_classes = [num_classes]
      elif label_type == 'single_multiclass' and isinstance(num_classes, list):
        num_classes = num_classes[0]
        
      subnetworks = [] 
      for i in range(len(input_datas)):
        with tf.variable_scope(scope, reuse=(i>0)):
          subnetwork = create_cnn_layers(network_type, input_datas[i], keep_probability)
          subnetwork = flatten(subnetwork)
          subnetworks.append(subnetwork)
          
      if len(subnetworks) > 1:
        network = merge(subnetworks, axis=1, mode='concat')
      else:
        network = subnetworks[0]
        
      if fc1_dimension > 0:
        network = fully_connected(network, fc1_dimension, activation='relu', restore=ifrestore, name=fc1_name)
        network = dropout(network, keep_probability, name=('dropout_' + fc1_name))
      if fc2_dimension > 0:
        network = fully_connected(network, fc2_dimension, activation='relu', restore=ifrestore, name=fc2_name)
    
    else:  # 'attribute_classification'
      # Create normal deep network for attribute classification
      if isinstance(input_datas, list):
        input_datas = input_datas[0]
      network = create_cnn_layers(network_type, input_datas, keep_probability)
    
  # Create the FC layers and regressors attaching to the convolutional layers
  network = create_fc_and_regressor_layers(network,
                                           num_classes,
                                           optimizer,
                                           loss_func,
                                           fine_tuning,
                                           label_type,
                                           lr_mult_fc,
                                           resume)
  return network
