# ***************
# deeplab_lfov.py
# ***************
# Tflearn re-implementation of DeepLab-LargeFOV model, based on the original TensorFlow
# implemenation in https://github.com/DrSleep/tensorflow-deeplab-lfov
# (Author:  Yu Chen).
# ***************

# ********************************************************
# Necessary Imports
import tensorflow as tf
from tflearn.layers.core import dropout
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.utils import get_incoming_shape

"""DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.

This mplements a multi-layer convolutional neural network for semantic image segmentation task.
This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
there for details.
"""
# --------------------------------------------------------
# The DeepLab-LargeFOV model can be represented as follows:
## input -> [conv-relu](dilation=1, channels=64) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=128) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=256) x 3 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=512) x 3 -> [max_pool](stride=1)
##       -> [conv-relu](dilation=2, channels=512) x 3 -> [max_pool](stride=1) -> [avg_pool](stride=1)
##       -> [conv-relu](dilation=12, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=20) -> [pixel-wise softmax loss].

def create_network(input_data,
                   retrain_final_layers = 0,
                   remove_final_layers = 0,
                   num_classes = 2,
                   keep_prob = 0.5,
                   kernel_size = 3):
                   
  # --------------------------------------------------------
  # All convolutional and pooling operations are applied using kernels of size 3x3; 
  # padding is added so that the output of the same size as the input.
  
  # VGG like architecture
  network = conv_2d(input_data, 64, 3, activation='relu', name='conv1_1')
  network = conv_2d(network, 64, 3, activation='relu', name='conv1_2')
  network = max_pool_2d(network, kernel_size, strides=2, name='pool1')

  network = conv_2d(network, 128, 3, activation='relu', name='conv2_1')
  network = conv_2d(network, 128, 3, activation='relu', name='conv2_2')
  network = max_pool_2d(network, kernel_size, strides=2, name='pool2')

  network = conv_2d(network, 256, 3, activation='relu', name='conv3_1')
  network = conv_2d(network, 256, 3, activation='relu', name='conv3_2')
  network = conv_2d(network, 256, 3, activation='relu', name='conv3_3')
  network = max_pool_2d(network, kernel_size, strides=2, name='pool3')

  network = conv_2d(network, 512, 3, activation='relu', name='conv4_1')
  network = conv_2d(network, 512, 3, activation='relu', name='conv4_2')
  network = conv_2d(network, 512, 3, activation='relu', name='conv4_3')
  network = max_pool_2d(network, kernel_size, strides=1, name='pool4')

  network = conv_2d(network, 512, 3, activation='relu', name='conv5_1', dilation=2)
  network = conv_2d(network, 512, 3, activation='relu', name='conv5_2', dilation=2)
  network = conv_2d(network, 512, 3, activation='relu', name='conv5_3', dilation=2)
  network = max_pool_2d(network, kernel_size, strides=1, name='pool5')
  network = avg_pool_2d(network, kernel_size, strides=1, name='pool5_avg')
  
  # Fully convoluational
  if remove_final_layers < 3:
    network = conv_2d(network, 1024, 3, activation='relu', name='fc6', dilation=12, restore=(retrain_final_layers < 3))
    network = dropout(network, keep_prob, name='dropout1')
  if remove_final_layers < 2:
    network = conv_2d(network, 1024, 1, activation='relu', name='fc7', restore=(retrain_final_layers < 2))
    network = dropout(network, keep_prob, name='dropout2')
  if remove_final_layers < 1:
    network = conv_2d(network, num_classes, 1, activation='linear', name='fc8_voc12', restore=(retrain_final_layers < 1))
  
  return network

  
def pixelwise_preds(raw_prediction, input_size, flag_soft_labels = False):
  """Create the network and run inference on the input batch.
  
  Args:
    prediction_output:  the raw output from the inference of deeplab network
    input_size: dimensions of pre-processed images.
    flag_soft_labels: False = No soft labels [Hard labels with argmax], True = Soft labels [Each pixel, we get prob over classes].
    
  Returns:
    Argmax over the predictions of the network of the same shape as the input.
  """
    
  output_shape = get_incoming_shape(raw_prediction)[1:3]
  if output_shape[0] != input_size[0] or output_shape[1] != input_size[1]:
    prediction_output = tf.image.resize_bilinear(raw_prediction, input_size)
  else:
    prediction_output = raw_prediction
  if not flag_soft_labels:
    prediction_output = tf.argmax(prediction_output, dimension=3)
    prediction_output = tf.expand_dims(prediction_output, dim=3) # Create 4D-tensor.
    return tf.cast(prediction_output, tf.uint8)
  return prediction_output
    


