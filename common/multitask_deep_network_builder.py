# ***************
# multitask_deep_network_builder.py
# ***************
# The generic builder module to create various deep network architectures for
#  deep learning training and testing. 
# (Author:  Yu Chen)
# ***************

# Import necessary external libraries 
import numpy as np
import tensorflow as tf
from tflearn.layers.conv import conv_2d, upsample_2d
from tflearn.layers.core import fully_connected, flatten, input_data, dropout
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge, merge_outputs
from tflearn.metrics import Accuracy, MultiLabelAccuracy
from tflearn.objectives import categorical_crossentropy, full_categorical_crossentropy, weighted_full_categorical_crossentropy
from tflearn.utils import get_incoming_shape

import sys
from common.conv_layer_factory import create_cnn_layers
from common.lr_multiplier_factory import create_lr_multipler_dictionary
from common.objectives import *
from common.optimizer_factory import create_optimizer
from common.triplet_block import create_triplet_block
from net.deeplab_lfov import pixelwise_preds


DEFAULT_SCOPE = ''

# create TensorFlow placeholders for the multiple-parallel data input 
def create_multitask_data_inputs(image_height, image_width, image_depth,
                                 data_counts, data_preprocessing, data_augmentation, data_scope = DEFAULT_SCOPE, force_reuse = False):
  if not isinstance(data_counts, list) or not isinstance(data_preprocessing, list) or not isinstance(data_augmentation, list):
    raise Exception('The argument must be lists')
                                
  if len(data_counts) != len(data_preprocessing) or len(data_counts) != len(data_augmentation):
    raise Exception('The argument have the same dimensionality!')
    
  with tf.variable_scope(data_scope) as scope_in:
    # create the data input placeholders
    data_inputs = [] 
    for h in range(len(data_counts)):
      if data_counts[h] <= 0:
        raise Exception('The data count must be positive!')
        
      for i in range(data_counts[h]):
        with tf.variable_scope(scope_in, reuse=(h > 0 or i > 0)):
          data_input = input_data(shape=[None, image_height, image_width, image_depth],
                                  data_preprocessing=data_preprocessing[h],
                                  data_augmentation=data_augmentation[h])
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
                                   lr_filter_string = "FullyConnected",
                                   is_training = False,
                                   landmark_dimension = 2,
                                   output_shape = None):
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
    if len(fc_outputs) > 1:
      network = merge(fc_outputs, axis=1, mode='concat')   
    else:
      network = fc_outputs[0]
    metric = MultiLabelAccuracy(k=len(num_classes), name='m_acc')
  elif label_type == 'single_multiclass': # Single multi-class label classification
    network = fully_connected(network, num_classes, activation='softmax',restore=ifrestore, name=lr_filter_string)
    metric = 'accuracy'
  elif label_type == 'single_binary': # Single binary label classification
    network = fully_connected(network, num_classes, activation='sigmoid',restore=ifrestore, name=lr_filter_string)
    metric = 'accuracy'
  elif label_type == 'multi_regression': # Multi-dimensional regression
    network = fully_connected(network, num_classes, activation='linear',restore=ifrestore, name=lr_filter_string)
    metric = 'r2'
  elif label_type == 'landmark_regression': # Landmark regression (binary visibility + 2d positions X,Y) 
    network_visibility = fully_connected(network, num_classes / (landmark_dimension + 1), activation='sigmoid',restore=ifrestore, name=lr_filter_string + 'Vis')
    network_position = fully_connected(network, num_classes * landmark_dimension / (landmark_dimension + 1), activation='linear',restore=ifrestore, name=lr_filter_string + 'Pos')
    network = merge([network_visibility, network_position], axis=1, mode='concat')   
    metric = MultiLabelAccuracy(k=num_classes / (landmark_dimension + 1), name='m_acc', binary=True)
  elif label_type == 'regressor_only': # Only build the regression layer
    metric = 'accuracy'
  elif label_type == 'segmentation':
    metric = Accuracy(one_hot_classes=num_classes)
  else:
    raise Exception('Unknown label type')
  
  lr_mult = None
  if fine_tuning:
    lr_mult = create_lr_multipler_dictionary([lr_filter_string], lr_mult_fc)
        
  if loss == None:
    loss = 'categorical_crossentropy'
    
  if is_training:     
    network = regression(network, optimizer=optimizer, loss=loss, metric=metric, restore=ifrestore, lr_multipliers=lr_mult, output_shape=output_shape)
  return network
    
# A factory to create deep NNs of different network architecture with
# different optimization parameters
def create_dnn(network_type,
               input_datas,  # a single or a list of input-data modules
               task_configs,
               optimizer_config = None,
               is_training = True,
               fine_tuning = False,
               lr_mult_fc = 10,
               resume = False,
               weight_sharing = 'all',
               data_scope = DEFAULT_SCOPE,
               force_reuse = False):
               
  # Create an optimizer from the specified JSON config file
  optimizer = "Momentum" if not optimizer_config else create_optimizer(optimizer_config)
  
  # Specify different network architectures. Normalize to lower cases
  network_type = network_type.lower()
  
  # Set keep probability for drop-out
  keep_probability = 0.6 if is_training else 1.0
  ifrestore = (not fine_tuning) or resume
  input_offset = 0
  scope_common = None
  networks = []
  with tf.variable_scope(data_scope, reuse=force_reuse) as scope_common:
    for h in range(len(task_configs)):
      task_type = str(task_configs[h]['task_type']) if task_configs[h].get('task_type') else 'classification'
      task_id = str(task_configs[h]['task_id']) if task_configs[h].get('task_id') else h
      label_type = str(task_configs[h]['problem_type']) if task_configs[h].get('problem_type') else None
      num_classes = task_configs[h]['num_classes'] if task_configs[h].get('num_classes') else 2
      fc1_dimension = task_configs[h]['fc1_dimension'] if task_configs[h].get('fc1_dimension') else 0
      fc2_dimension = task_configs[h]['fc2_dimension'] if task_configs[h].get('fc2_dimension') else 0
      fc1_name = str(task_configs[h]['fc1_name']) if task_configs[h].get('fc1_name') else "Task%s_FullyConnected1" % (task_id)
      fc2_name = str(task_configs[h]['fc2_name']) if task_configs[h].get('fc2_name') else "Task%s_FullyConnected2" % (task_id)
      fc1_activation = str(task_configs[h]['fc1_activation']) if task_configs[h].get('fc1_activation') else 'relu'
      fc2_activation = str(task_configs[h]['fc2_activation']) if task_configs[h].get('fc2_activation') else 'relu'
      fc_name_filter = str(task_configs[h]['fc_name_filter']) if task_configs[h].get('fc_name_filter') else "Task%s_FullyConnected" % (task_id)
      conv_lr_filter = str(task_configs[h]['conv_lr_filter']) if task_configs[h].get('conv_lr_filter') else ''
      landmark_weight = task_configs[h]['landmark_weight'] if task_configs[h].get('landmark_weight') else 0.001
      landmark_dimension = task_configs[h]['landmark_dimension'] if task_configs[h].get('landmark_dimension') else 2
      mapped_num_classes = task_configs[h]['mapped_num_classes'] if task_configs[h].get('mapped_num_classes') else num_classes
      retrain_final_layers = task_configs[h]['retrain_final_layers'] if task_configs[h].get('retrain_final_layers') else 0
      remove_final_layers = task_configs[h]['remove_final_layers'] if task_configs[h].get('remove_final_layers') else 0
      generate_soft_labels = (str(task_configs[h]['generate_soft_labels']).lower() == "true") if task_configs[h].get('generate_soft_labels') else False
      label_downsize_factor = task_configs[h]['label_downsize_factor'] if task_configs[h].get('label_downsize_factor') else 1
 
      # Configure the loss function of the task
      loss_func = str(task_configs[h]['loss_func']) if task_configs[h].get('loss_func') else "categorical_crossentropy"
      loss_weight = task_configs[h]['loss_weight'] if task_configs[h].get('loss_weight') else 1
      if loss_func == "categorical_crossentropy" and loss_weight != 1 and loss_weight > 0:
        loss_func = lambda y_pred, y_true: categorical_crossentropy(y_pred, y_true) * loss_weight
      elif loss_func == "full_categorical_crossentropy" and loss_weight != 1 and loss_weight > 0:
        loss_func = lambda y_pred, y_true: full_categorical_crossentropy(y_pred, y_true) * loss_weight
      elif loss_func == 'weighted_full_categorical_crossentropy': 
        negative_sample_weight = task_configs[h]['negative_sample_weight'] if task_configs[h].get('negative_sample_weight') else 1
        loss_func = lambda y_pred, y_true: weighted_full_categorical_crossentropy(y_pred, y_true, 1, negative_sample_weight)
      elif loss_func == 'landmark_visibility_loss':
        num_landmarks = num_classes / (landmark_dimension + 1)
        loss_func = lambda y_pred, y_true: landmark_visibility_loss(y_pred, y_true,
                                                                    alpha=landmark_weight,
                                                                    num_landmarks=num_landmarks,
                                                                    total_classes=num_classes)
      elif loss_func == 'landmark_visibility_mask_loss':
        num_landmarks = num_classes / (landmark_dimension + 1)
        loss_func = lambda y_pred, y_true: landmark_visibility_mask_loss(y_pred, y_true,
                                                                         alpha=landmark_weight,
                                                                         num_landmarks=num_landmarks,
                                                                         total_classes=num_classes)
      elif loss_func == 'pixelwise_softmax_loss':
        loss_func = lambda y_pred, y_true: pixelwise_softmax_loss(y_pred, y_true,
                                                                  num_classes=num_classes)

      network = None
      estimator_output_shape = None
      if task_type == 'triplet':
        # create triplet network for similarity metric learning
        network = create_triplet_block(network_type = network_type,
                                       input_data1 = input_datas[input_offset],
                                       input_data2 = input_datas[input_offset + 1],
                                       input_data3 = input_datas[input_offset + 2],
                                       weight_sharing = weight_sharing, 
                                       fc1_dimension = fc1_dimension,
                                       fc2_dimension = fc2_dimension,
                                       fc1_name = fc1_name,
                                       fc2_name = fc2_name,
                                       keep_probability = keep_probability,
                                       ifrestore = ifrestore,
                                       scope_enforced = scope_common,
                                       reuse_first_cnn = (force_reuse or h > 0))
        label_type = 'regressor_only'
        input_offset = input_offset + 3
        
      elif task_type == 'segmentation':
        # Create a segmentation network.
        if network_type != 'deeplab_lfov' and network_type != 'deeplab_lfov_upsample':
          raise Exception('Network type %s is not supported for segmentation tasks.' % (network_type))
          
        with tf.variable_scope(scope_common, reuse=(force_reuse or (h > 0))):
          network = create_cnn_layers(network_type, input_datas[input_offset], \
                                      keep_probability, retrain_final_layers=retrain_final_layers,
                                      num_classes=mapped_num_classes, remove_final_layers=remove_final_layers)

        # Additional 1x1 Conv/Pixel-wise FC layers
        with tf.variable_scope(scope_common, reuse=(force_reuse or ((h > 0) if (weight_sharing == 'all' or weight_sharing == 'fc1') else False))):
          if fc1_dimension > 0:
            if network_type == 'deeplab_lfov_upsample' and remove_final_layers > 1:
              network = upsample_2d(network, 2, name='upsample' + fc1_name)
              network = conv_2d(network, fc1_dimension, 3, activation=fc1_activation, restore=ifrestore, name=fc1_name)
              network = conv_2d(network, fc1_dimension / 2, 1, activation=fc1_activation, restore=ifrestore, name=fc1_name + '_1')
            else:
              network = conv_2d(network, fc1_dimension, 1, activation=fc1_activation, restore=ifrestore, name=fc1_name)
            network = dropout(network, keep_probability, name=('dropout_' + fc1_name))
            
        with tf.variable_scope(scope_common, reuse=(force_reuse or ((h > 0) if weight_sharing == 'all' else False))):
          if fc2_dimension > 0:
            if network_type == 'deeplab_lfov_upsample' and remove_final_layers > 0:
              network = upsample_2d(network, 2, name='upsample' + fc2_name)
              network = conv_2d(network, fc2_dimension, 3, activation=fc2_activation, restore=ifrestore, name=fc2_name)
            else:
              network = conv_2d(network, fc2_dimension, 1, activation=fc2_activation, restore=ifrestore, name=fc2_name)

        input_shape = get_incoming_shape(input_datas[input_offset])
        output_shape = get_incoming_shape(network)
        print input_shape, output_shape
        estimator_output_shape = [output_shape[0], int(input_shape[1] / label_downsize_factor), int(input_shape[2] / label_downsize_factor), 1]

        label_type = 'segmentation'
        input_offset = input_offset + 1
        
      elif task_type == 'segmentation_inference':
        # Create a segmentation network.
        if network_type != 'deeplab_lfov' and network_type != 'deeplab_lfov_upsample':
          raise Exception('Network type %s is not supported for segmentation tasks.' % (network_type))

        feature_input_offset = task_configs[h]['input_offset'] if task_configs[h].get('input_offset') else 0
        
        with tf.variable_scope(scope_common, reuse=(force_reuse or (h > 0))):
          network = create_cnn_layers(network_type, input_datas[feature_input_offset], \
                                      keep_probability, retrain_final_layers=0,
                                      num_classes=mapped_num_classes, remove_final_layers=remove_final_layers)
                                      
        # Additional 1x1 Conv/Pixel-wise FC layers
        with tf.variable_scope(scope_common, reuse=(force_reuse or ((h > 0) if (weight_sharing == 'all' or weight_sharing == 'fc1') else False))):
          if fc1_dimension > 0:
            if network_type == 'deeplab_lfov_upsample':
              network = upsample_2d(network, 2, name='upsample' + fc1_name)
              network = conv_2d(network, fc1_dimension, 3, activation=fc1_activation, restore=ifrestore, name=fc1_name)
              network = conv_2d(network, fc1_dimension / 2, 1, activation=fc1_activation, restore=ifrestore, name=fc1_name + '_1')
            else:
              network = conv_2d(network, fc1_dimension, 1, activation=fc1_activation, restore=ifrestore, name=fc1_name)
           
        with tf.variable_scope(scope_common, reuse=(force_reuse or ((h > 0) if weight_sharing == 'all' else False))):
          if fc2_dimension > 0:
            if network_type == 'deeplab_lfov_upsample':
              network = upsample_2d(network, 2, name='upsample' + fc2_name)
              network = conv_2d(network, fc2_dimension, 3, activation=fc2_activation, restore=ifrestore, name=fc2_name)
            else:
              network = conv_2d(network, fc2_dimension, 1, activation=fc2_activation, restore=ifrestore, name=fc2_name)

        input_shape = get_incoming_shape(input_datas[feature_input_offset])

        network = pixelwise_preds(network, input_shape[1:3], generate_soft_labels)
        
        label_type = None
        
      elif task_type == 'feature_extraction':
        # Use the pre-trained attribute classification or similarity metric network for feature extraction
        if is_training:
          raise Exception('feature extraction networks can only be used in the testing stage.')
        #if isinstance(input_datas, list):
        #  input_datas = input_datas[0]
        feature_input_offset = task_configs[h]['input_offset'] if task_configs[h].get('input_offset') else 0
        num_views = task_configs[h]['num_views'] if task_configs[h].get('num_views') else 1
        
        subnetworks = []
        for i in range(num_views):
          with tf.variable_scope(scope_common, reuse=(force_reuse or h > 0 or i > 0)):
            subnetwork = create_cnn_layers(network_type, input_datas[feature_input_offset + i], keep_probability)
            subnetwork = flatten(subnetwork)
            subnetworks.append(subnetwork)
            
        with tf.variable_scope(scope_common, reuse=(force_reuse or ((h > 0) if weight_sharing else False))):
          if len(subnetworks) > 1:
            network = merge(subnetworks, axis=1, mode='concat')
          else:
            network = subnetworks[0]

        # Add a couple more FC layers for similarity metric embedding
        with tf.variable_scope(scope_common, reuse=(force_reuse or ((h > 0) if (weight_sharing == 'all' or weight_sharing == 'fc1') else False))):
          if fc1_dimension > 0:
            network = fully_connected(network, fc1_dimension, activation=fc1_activation, restore=ifrestore, name=fc1_name)
            
        with tf.variable_scope(scope_common, reuse=(force_reuse or ((h > 0) if weight_sharing == 'all' else False))):
          if fc2_dimension > 0:
            network = fully_connected(network, fc2_dimension, activation=fc2_activation, restore=ifrestore, name=fc2_name)
            
      elif task_type == 'classification' or task_type == 'landmark_prediction':
        # Create normal deep network for attribute classification
        num_views = task_configs[h]['num_views'] if task_configs[h].get('num_views') else 1
        
        # Regularize the input
        if label_type == 'multi_multiclass' and not isinstance(num_classes, list):
          num_classes = [num_classes]
        elif (label_type == 'single_multiclass' or label_type == 'landmark_regression') and isinstance(num_classes, list):
          num_classes = num_classes[0]
        
        subnetworks = []
        for i in range(num_views):
          with tf.variable_scope(scope_common, reuse=(force_reuse or h > 0 or i > 0)):
            subnetwork = create_cnn_layers(network_type, input_datas[input_offset + i], keep_probability)
            subnetwork = flatten(subnetwork)
            subnetworks.append(subnetwork)
            
        with tf.variable_scope(scope_common, reuse=(force_reuse or ((h > 0) if weight_sharing else False))):
          if len(subnetworks) > 1:
            network = merge(subnetworks, axis=1, mode='concat')
          else:
            network = subnetworks[0]
        
        with tf.variable_scope(scope_common, reuse=(force_reuse or ((h > 0) if (weight_sharing == 'all' or weight_sharing == 'fc1') else False))):
          if fc1_dimension > 0:
            network = fully_connected(network, fc1_dimension, activation=fc1_activation, restore=ifrestore, name=fc1_name)
            network = dropout(network, keep_probability, name=('dropout_' + fc1_name))

        with tf.variable_scope(scope_common, reuse=(force_reuse or ((h > 0) if weight_sharing == 'all' else False))):
          if fc2_dimension > 0:
            network = fully_connected(network, fc2_dimension, activation=fc2_activation, restore=ifrestore, name=fc2_name)
        input_offset = input_offset + num_views
      else:
        raise Exception('Unsupported task type: ', task_type)
          
      # Create the FC layers and regressors attaching to the convolutional layers
      network_fc_regressors = create_fc_and_regressor_layers(network,
                                                             num_classes,
                                                             optimizer,
                                                             loss_func,
                                                             fine_tuning,
                                                             label_type,
                                                             lr_mult_fc,
                                                             resume,
                                                             fc_name_filter,
                                                             is_training,
                                                             landmark_dimension,
                                                             estimator_output_shape)
      networks.append(network_fc_regressors)
  if len(networks) > 1:
    sample_output_shape = get_incoming_shape(networks[0])
    final_network = merge(networks, axis=len(sample_output_shape) - 1, mode='concat')
  else:
    final_network = networks[0]
  return final_network
