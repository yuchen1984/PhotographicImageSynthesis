# ***************
# triplet_block.py
# ***************
# Generate network architectures for triplet similarity learning 
# (Author:  Yu Chen)
# ***************

# Import necessary external libraries 
import numpy as np
import tensorflow as tf
from tflearn.config import _EPSILON,_FLOATX
from tflearn.layers.core import fully_connected, activation, flatten, dropout
from tflearn.layers.merge_ops import merge
from tflearn.utils import get_incoming_shape

import sys
from common.conv_layer_factory import create_cnn_layers

# Create a weight-sharing network architecture for triplet similarity metric learning
# It return the network block and the TF scope for further weight-sharing
def create_triplet_block(network_type,
                         input_data1,
                         input_data2,
                         input_data3,
                         scope_enforced,
                         reuse_first_cnn = False,
                         weight_sharing = 'all', 
                         fc1_dimension = 2048,
                         fc2_dimension = 512,
                         fc1_name = 'FullyConnected1',
                         fc2_name = 'FullyConnected2',
                         keep_probability = 0.5,
                         use_combined_loss = False,
                         ifrestore = False):
    # create triplet network for similarity metric learning
    if input_data1 == None or input_data2 == None or input_data3 == None:
      raise Exception('input_data1, input_data2 and input_data3 must not be None to train a Triplet network')
    if fc1_dimension <= 0:
      raise Exception('The first fully connected layer may have a positive dimensionality!')
      
    fc1_name_alternative = fc1_name if weight_sharing != None else (fc1_name + 's')
    fc2_name_alternative = fc2_name if weight_sharing == 'all' else (fc2_name + 's')
 
    with tf.variable_scope(scope_enforced, reuse=reuse_first_cnn):
      network1 = create_cnn_layers(network_type, input_data1, keep_probability, False)
      network1 = flatten(network1)        
      network1 = fully_connected(network1, fc1_dimension, activation=('relu' if fc2_dimension > 0 else 'linear'),restore=ifrestore, name=fc1_name)
      if fc2_dimension > 0:
        network1 = fully_connected(network1, fc2_dimension, activation='linear',restore=ifrestore, name=fc2_name)
    
    with tf.variable_scope(scope_enforced, reuse=True):
      network2 = create_cnn_layers(network_type, input_data2, keep_probability, False)
      network2 = flatten(network2)        
      network2 = fully_connected(network2, fc1_dimension, activation=('relu' if fc2_dimension > 0 else 'linear'),restore=ifrestore, name=fc1_name_alternative)
      if fc2_dimension > 0:
        network2 = fully_connected(network2, fc2_dimension, activation='linear',restore=ifrestore, name=fc2_name_alternative)

    with tf.variable_scope(scope_enforced, reuse=True):
      network3 = create_cnn_layers(network_type, input_data3, keep_probability, False)
      network3 = flatten(network3)        
      network3 = fully_connected(network3, fc1_dimension, activation=('relu' if fc2_dimension > 0 else 'linear'),restore=ifrestore, name=fc1_name_alternative)
      if fc2_dimension > 0:
        network3 = fully_connected(network3, fc2_dimension, activation='linear',restore=ifrestore, name=fc2_name_alternative)
    # Compute cosine similarity and apply thresholding
    print 'Incoming feature dimension: ', get_incoming_shape(network1), get_incoming_shape(network2), get_incoming_shape(network3)
    
    # using cosine distance / correlation as the similarity metric
    dot_product_positive = tf.reduce_sum(tf.mul(network1, network2), reduction_indices=len(network1.get_shape()) - 1, keep_dims=True)
    dot_product_negative = tf.reduce_sum(tf.mul(network1, network3), reduction_indices=len(network1.get_shape()) - 1, keep_dims=True)
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(network1), reduction_indices=len(network1.get_shape()) - 1, keep_dims=True))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(network2), reduction_indices=len(network2.get_shape()) - 1, keep_dims=True))
    norm3 = tf.sqrt(tf.reduce_sum(tf.square(network3), reduction_indices=len(network3.get_shape()) - 1, keep_dims=True))
    
    similarity_positive = tf.div(dot_product_positive, tf.maximum(tf.mul(norm1, norm2), tf.cast(_EPSILON, dtype=_FLOATX)))
    similarity_negative = tf.div(dot_product_negative, tf.maximum(tf.mul(norm1, norm3), tf.cast(_EPSILON, dtype=_FLOATX)))
    
    # Define Triple loss based on the difference between the positive pair similarity and negative pair similarity
    triplet_loss = tf.clip_by_value(similarity_positive - similarity_negative, tf.cast(_EPSILON, dtype=_FLOATX), tf.cast(1.-_EPSILON, dtype=_FLOATX))
    if use_combined_loss:  # combine_loss = similarity_loss * triplet_loss
      similarity_loss = tf.clip_by_value(similarity_positive, tf.cast(_EPSILON, dtype=_FLOATX), tf.cast(1.-_EPSILON, dtype=_FLOATX))
      combined_loss = tf.mul(triplet_loss, similarity_loss)
      network = merge([combined_loss, 1.0 - combined_loss], axis=1, mode='concat')
    else:
      network = merge([triplet_loss, 1.0 - triplet_loss], axis=1, mode='concat')
    return network
