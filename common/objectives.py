# ***************************************************************************************************
# ***************************************************************************************************
# New Loss Objectives for CNN training
# -----------------------------------------
# DETAILS:
# Contains definitions of new loss objectives
# (Author:  Yu Chen, Sukrit Shankar)
# -----------------------------------------
# ***************************************************************************************************
# ***************************************************************************************************

# ********************************************************
# Necessary Imports - Generic
import tensorflow as tf
import tflearn
from tflearn.objectives import *
from tflearn.utils import get_incoming_shape

# ********************************************************
# Necessary Imports - Self Defined

"""
-------------------------------
Function Name: landmark_visibility_loss
Function Role: Returns the loss considering the landmark visibility
Arguments:
    y_pred, y_true
Returns (Populates):
    loss
Remarks:
    Each landmark is a 2D point, and visibility is a binary label
    Expects y_pred and y_true as [vis_1, vis_2, ...., land_x_1, land_y_1, land_x_2, land_y_2, ...]
"""

def landmark_visibility_loss(y_pred, y_true, alpha, num_landmarks, total_classes):
    L_visibility = full_categorical_crossentropy(y_pred[:,0:num_landmarks], y_true[:,0:num_landmarks])
    L_landmarks = mean_square(y_pred[:,num_landmarks:total_classes], y_true[:,num_landmarks:total_classes])
    return tf.add(L_visibility, alpha * L_landmarks)


"""
-------------------------------
Function Name: landmark_visibility_mask_loss
Function Role: Returns the loss considering the landmark visibility while masking the positions of invisible ones
Arguments:
    y_pred, y_true
Returns (Populates):
    loss
Remarks:
    Each landmark is a 2D point, and visibility is a binary label
    Expects y_pred and y_true as [vis_1, vis_2, ...., land_x_1, land_y_1, land_x_2, land_y_2, ...]
"""

def landmark_visibility_mask_loss(y_pred, y_true, alpha, num_landmarks, total_classes):
    landmark_dimension = total_classes / num_landmarks - 1
    
    L_visibility = full_categorical_crossentropy(y_pred[:,0:num_landmarks], y_true[:,0:num_landmarks])

    L_landmarks = tf.multiply(tf.square(y_pred[:, num_landmarks] - y_true[:,num_landmarks]), y_true[:,0])
    for j in range(1, landmark_dimension):
      L_landmarks = tf.add(L_landmarks,tf.multiply(tf.square(y_pred[:, num_landmarks + j] - y_true[:, num_landmarks + j]), y_true[:,0]))
    for i in range(1, num_landmarks):
      for j in range(landmark_dimension):
        L_landmarks = tf.add(L_landmarks,tf.multiply(tf.square(y_pred[:, num_landmarks + landmark_dimension * i + j] - y_true[:,num_landmarks + landmark_dimension * i + j]), y_true[:,i]))

    L_landmarks = tf.reduce_mean(tf.divide(L_landmarks, (total_classes - num_landmarks)))
    return tf.add(L_visibility, alpha * L_landmarks)

    
def prepare_label(input_batch, new_size, depth, one_hot = False):
  """Resize masks and perform one-hot encoding.

  Args:
    input_batch: input tensor of shape [batch_size H W 1].
    new_size: a tensor with new height and width.

  Returns:
    Outputs a tensor of shape [batch_size h w 21]
    with last dimension comprised of 0's and 1's only.
  """
  with tf.name_scope('label_encode'):
    input_shape = get_incoming_shape(input_batch)
    if input_shape[1] != new_size[0] or input_shape[2] != new_size[1]:
      input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # As labels are integer numbers, need to use NN interp.
    if one_hot:
      input_batch = tf.cast(input_batch, tf.uint8)
      input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # Reducing the channel dimension.
      input_batch = tf.one_hot(input_batch, depth)
  return input_batch

      
def pixelwise_softmax_loss(y_pred, y_true, num_classes):
  """Create the network, run inference on the input batch and compute loss.
  
  Args:
    y_pred: batch of prediction results.
    y_true: batch of ground truth label images.
    num_classes:  number of classes defined for the segmentation problem
    
  Returns:
    Pixel-wise softmax loss.
  """
  prediction = tf.reshape(y_pred, [-1, num_classes])
  # Need to resize labels and convert using one-hot encoding.
  gt = prepare_label(y_true, tf.pack(y_pred.get_shape()[1:3]), num_classes, True)
  gt = tf.reshape(gt, [-1, num_classes])
  
  # Pixel-wise softmax loss.
  loss = tf.nn.softmax_cross_entropy_with_logits(prediction, gt)
  reduced_loss = tf.reduce_mean(loss)
  
  return reduced_loss
    




































