# ***************
# lr_multiplier_factory.py
# ***************
# The generic factory module to create a dictionary of learning_rate multiplers
# for a set of layers/variables with a name filter.
# (Author:  Yu Chen)
# ***************

import tensorflow as tf

  # Create a dictionary for learning rate multiplier
def create_lr_multipler_dictionary(lr_filter_string_list, lr_multiplier):
  all_vars = tf.trainable_variables()
  lr_mult_dict = dict()
  for v in all_vars:
    filter_flag = False
    for lr_filter_string in lr_filter_string_list:
      if lr_filter_string in v.op.name:
        filter_flag = True
        break
    
    if filter_flag:
      print v.op.name
      lr_mult_dict[v.op.name] = lr_multiplier
      
  return lr_mult_dict
