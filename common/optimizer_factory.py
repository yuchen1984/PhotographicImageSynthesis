# ***************
# optimizer_factory.py
# ***************
# The generic factory module to loading different optimizers from a JSON config file. 
# (Author:  Yu Chen)
# ***************

import json
import os.path
from tflearn.optimizers import *

def create_optimizer(config_file_path = None):
  if not os.path.isfile(config_file_path):
    raise Exception(config_file_path + ' does not exist!')
    
  with open(config_file_path, 'r') as fr:
    config = json.load(fr)

  optimization_method = "Momemtum"  # Set default optimization method as "Momemtum"
  
  optimizer_type = config['optimizer'].lower()

  initial_lr = config['initial_lr'] if config.get('initial_lr') else 0.001
  lr_decay = config['lr_decay'] if config.get('lr_decay') else 0.9
  decay_step = config['decay_step'] if config.get('decay_step') else 1000
  
  if (optimizer_type == 'momentum'):
    momentum = config['momentum'] if config.get('momentum') else 0.9
    staircase = config['staircase'] if config.get('staircase') else False
    print "Momentum: initial_lr=", initial_lr, "lr_decay=", lr_decay, "decay_step=", decay_step, "staircase=", staircase
    optimization_method = Momentum(learning_rate=initial_lr, lr_decay=lr_decay, decay_step=decay_step, momentum=momentum, staircase=staircase)
  elif (optimizer_type == 'sgd'):
    staircase = config['staircase'] if config.get('staircase') else False
    optimization_method = SGD(learning_rate=initial_lr, lr_decay=lr_decay, decay_step=decay_step, staircase=staircase)
  elif (optimizer_type == 'rmsprop'):
    momentum = config['momentum'] if config.get('momentum') else 0.0
    optimization_method = SGD(learning_rate=initial_lr, decay=lr_decay, momentum=momentum)
  elif (optimizer_type == 'adam'):
    beta1 = config['lr_beta1'] if config.get('lr_beta1') else 0.9
    beta2 = config['lr_beta2'] if config.get('lr_beta2') else 0.99
    optimization_method = Adam(learning_rate=initial_lr, beta1=beta1, beta2=beta2)
  elif (optimizer_type == 'adagrad'):
    initial_accumulator_value = config['initial_accumulator_value'] if config.get('initial_accumulator_value') else 0.1
    optimization_method = AdaGrad(learning_rate=initial_lr, initial_accumulator_value=initial_accumulator_value)
  else:
    raise Exception('Unknown optimisation method!')
  
  return optimization_method
