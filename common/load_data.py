# ***************
# load_data.py
# ***************
# Modules of functions for loading data to train and test with TFLearn
# (Author : Sukrit Shankar and Yu Chen)
# ***************

# Necessary Imports
from __future__ import division, absolute_import

import os
import numpy as np
import scipy 
import scipy.io

import tflearn
from tflearn.data_utils import *

class LabelImagePreloader(Preloader):
    def __init__(self, array, image_shape, n_class, categorical=True, shared_image_dict = None):
        fn = lambda x: self.preload(x, image_shape, n_class, categorical, shared_image_dict)
        super(LabelImagePreloader, self).__init__(array, fn)

    # NB: An optional shared image dictionary is supported to avoid memory leakage caused by repetitive image opening. 
    def preload(self, path, image_shape, n_class, categorical=True, shared_image_dict = None):
        if shared_image_dict and (path in shared_image_dict):
            img = shared_image_dict[path]
        else:      
            img = load_image(path)
            width, height = img.size
            if width != image_shape[0] or height != image_shape[1]:
                img = resize_image(img, image_shape[0], image_shape[1], resize_mode=Image.NEAREST)
            img.load()
            img = np.asarray(img, dtype='uint8')
            img_reformat = []
            if categorical:
              for i in range(img.shape[0]):
                img_reformat.append(to_categorical(img[i,:], n_class))
              img = np.asarray(img_reformat, dtype='float32')
            else:
              img = img.reshape(img.shape[0], img.shape[1], 1)
            if shared_image_dict:
                shared_image_dict[path] = img
        return img

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function to load single label data
def load_single_label_data(file_name, root_image_path, image_shape,
                           normalize = True, grayscale = False,
                           files_extension = None, filter_channel = False,
                           shared_image_dict = None, alpha_channel_only = False, splitter = ' '):
  with open(file_name, 'r') as f:
    images, labels = [], []
    for l in f.readlines():
      l = l.strip('\n').split(splitter)
      if not files_extension or any(flag in l[0] for flag in files_extension):
        l[0] = os.path.join(root_image_path, l[0])
        if filter_channel:
          if get_img_channel(l[0]) != 3:
            continue
        images.append(l[0])
        labels.append(int(l[1]))

  n_classes = np.max(labels) + 1
  labels_cat = to_categorical(labels, None)  # From List to List 
  X = ImagePreloader(images, image_shape, normalize, grayscale, shared_image_dict, alpha_channel_only)
  Y = LabelPreloader(labels_cat, n_classes, False)
  
  # Return X, Y and number of output classes 
  return X, Y, n_classes

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function to load segmentation label data
# The label file will be in the form of (input image file path, label image file path)
def load_segmentation_label_data(file_name, root_image_path, image_shape, n_classes,
                                 normalize = True, grayscale = False, categorical=True,
                                 files_extension = None, filter_channel = False, label_downsize_factor = 1,
                                 shared_image_dict = None, splitter = ' '):
  with open(file_name, 'r') as f:
    images, labels = [], []
    for l in f.readlines():
      l = l.strip('\n').split(splitter)
      if not files_extension or any(flag in l[0] for flag in files_extension):
        l[0] = os.path.join(root_image_path, l[0])
        l[1] = os.path.join(root_image_path, l[1])
        if filter_channel:
          if get_img_channel(l[0]) != 3 and get_img_channel(l[1]) != 1:
            continue
        images.append(l[0])
        labels.append(l[1])
  label_image_shape = (int(image_shape[0] / label_downsize_factor), int(image_shape[1] / label_downsize_factor))
  X = ImagePreloader(images, image_shape, normalize, grayscale, shared_image_dict, False)
  Y = LabelImagePreloader(labels, label_image_shape, n_classes, categorical, shared_image_dict)
  
  # Return X, Y
  return X, Y 
  
  
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function to load binary similarity data in the format of
# (Image 1, Image 2, 0/1). 1 for similar, 0 for dissimilar
def load_binary_similarity_data(file_name, root_image_path, image_shape,
                                normalize=True, grayscale = False,
                                files_extension = None, filter_channel = False, mirror_pairs = False,
                                shared_image_dict = None, alpha_channel_only = False, splitter = ' '):
  with open(file_name, 'r') as f:
    images1 = []
    images2 = []
    labels = []
    for l in f.readlines():
      l = l.strip('\n').split(splitter) 
      if not files_extension or any(flag in l[0] for flag in files_extension):
        l[0] = os.path.join(root_image_path, l[0])
        l[1] = os.path.join(root_image_path, l[1])
        if filter_channel:
          if get_img_channel(l[0]) != 3 or get_img_channel(l[1]) != 3:
            continue
        images1.append(l[0])
        images2.append(l[1])
        #labels.append([int(l[2])])
        labels.append([int(l[2]), 1 - int(l[2])])
        
        if mirror_pairs:
          # generate mirror similarity pairs for training/testing
          images1.append(l[1])
          images2.append(l[0])
          #labels.append([int(l[2])])
          labels.append([int(l[2]), 1 - int(l[2])])

  X1 = ImagePreloader(images1, image_shape, normalize, grayscale, shared_image_dict, alpha_channel_only)
  X2 = ImagePreloader(images2, image_shape, normalize, grayscale, shared_image_dict, alpha_channel_only)
  Y = LabelPreloader(labels, 2, False)
  
  return X1, X2, Y

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function to load triplet similarity data in the format of
# (Image 1, Image 2, Image 3, 0/1). 1 for (1,2) are more similar than (1,3), 0 for (1,3) are more similar than (1,2)
def load_triplet_similarity_data(file_name, root_image_path, image_shape,
                                normalize=True, grayscale = False,
                                files_extension = None, filter_channel = False, mirror_pairs = False,
                                shared_image_dict = None, alpha_channel_only = False, splitter = ' '):
  with open(file_name, 'r') as f:
    images1 = []
    images2 = []
    images3 = []
    labels = []
    for l in f.readlines():
      l = l.strip('\n').split(splitter) 
      if not files_extension or any(flag in l[0] for flag in files_extension):
        l[0] = os.path.join(root_image_path, l[0])
        l[1] = os.path.join(root_image_path, l[1])
        l[2] = os.path.join(root_image_path, l[2])
        if filter_channel:
          if get_img_channel(l[0]) != 3 or get_img_channel(l[1]) != 3 or get_img_channel(l[2]) != 3:
            continue
        images1.append(l[0])
        images2.append(l[1])
        images3.append(l[2])
        #labels.append([int(l[3])])
        labels.append([int(l[3]), 1 - int(l[3])])
        
        if mirror_pairs:
          # generate mirror similarity pairs for training/testing
          images1.append(l[1])
          images2.append(l[0])
          images3.append(l[2])
          #labels.append([int(l[3])])
          labels.append([int(l[3]), 1 - int(l[3])])

  X1 = ImagePreloader(images1, image_shape, normalize, grayscale, shared_image_dict, alpha_channel_only)
  X2 = ImagePreloader(images2, image_shape, normalize, grayscale, shared_image_dict, alpha_channel_only)
  X3 = ImagePreloader(images3, image_shape, normalize, grayscale, shared_image_dict, alpha_channel_only)
  Y = LabelPreloader(labels, 2, False)
  
  return X1, X2, X3, Y

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function to load multi dimensional data
def load_multi_dimensional_data(file_name, root_image_path, image_shape,
                                normalize=True, grayscale=False,
                                files_extension = None, filter_channel = False, headerSkip = 1,
                                shared_image_dict = None, binary_data = False, alpha_channel_only = False, splitter = ' '):
  with open(file_name, 'r') as f:
    count = 0
    lines = f.readlines()
    for l in lines:
      l = l.strip('\n').strip('\r').split(splitter)
      if count < headerSkip:
        count = count + 1
        continue
        
      if (count == headerSkip):
        n_classes = len(l) - 1
      
      if not files_extension or any(flag in l[0] for flag in files_extension):
        l[0] = os.path.join(root_image_path, l[0])
        if filter_channel:
          if get_img_channel(l[0]) != 3:
            continue
        count = count + 1

  n_entries = count - headerSkip

  with open(file_name, 'r') as f:
    images = []
    labels = np.zeros((n_entries,n_classes))
    count = 0
    for l in f.readlines():
      l = l.strip('\n').strip('\r').split(splitter) 
      if count < headerSkip:
        count = count + 1
        continue
      if not files_extension or any(flag in l[0] for flag in files_extension):
        l[0] = os.path.join(root_image_path, l[0])
        if filter_channel:
          if get_img_channel(l[0]) != 3:
            continue
        images.append(l[0])

        for i in range(1,len(l)):
          labels[count - headerSkip,i-1] = (float(l[i]) + 1) / 2.0 if binary_data else float(l[i])
        count = count + 1

  X = ImagePreloader(images, image_shape, normalize, grayscale, shared_image_dict, alpha_channel_only)
  Y = LabelPreloader(labels, n_classes, False)
  
  print "#Data = ", n_entries
  # Return X, Y and number of output classes 
  return X, Y, n_classes

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function to load multi multiclass label data
def load_multi_label_data(target_paths, image_shape, n_classes,
                          files_extension=None, filter_channel=False,
                          normalize=True, grayscale=False, categorical=True,
                          shared_image_dict = None, root_image_path = None, alpha_channel_only = False, splitter = ' '):
  images, labels = [], []
  for i in range(len(target_paths)):
    labelidx = []
    with open(target_paths[i], 'r') as f:
      for l in f.readlines():
        l = l.strip('\n').split(splitter)
        if not files_extension or any(flag in l(0) for flag in files_extension):
          if root_image_path:
            l[0] = os.path.join(root_image_path, l[0])
          if filter_channel:
            if get_img_channel(l[0]) != 3:
              continue
          if i == 0:
            images.append(l[0])
          labelidx.append(int(l[1]))

    if i == 0:
      labels = to_categorical(labelidx, n_classes[i]) if categorical else np.array([labelidx]).transpose()
    else:
      labels = np.concatenate((labels, (to_categorical(labelidx, n_classes[i])) if categorical else np.array([labelidx]).transpose()), axis = 1)
  print labels.shape
                    
  X = ImagePreloader(images, image_shape, normalize, grayscale, shared_image_dict, alpha_channel_only)
  Y = LabelPreloader(labels, n_classes, False)
  
  # Return X, Y
  return X, Y

# The helper function to load class definitions of labels from text files
# input:
#   label_files: a list of file names containing the definition of multiple class labels
#                e.g. ["main_category.txt", "garment-subcategory.txt", "pattern.txt"]
#   label_definition_dir: the directory that contains all the label definition files
# return:  
#   num_classes: an array of number of classes for each label definition
#                [10, 69, 20]
#   class_labels: an array of all defined labels names for each label definition
#                 e.g. [ ["Dress", "Skirt", "Trousers", ...], ["Shift-Dress", ...], ... ]
def load_class_labels(label_files, label_definition_dir, splitter = ','):
  label_type_count = len(label_files)

  class_labels = []
  num_classes = []
  for i in range(label_type_count):
    label_path = os.path.join(label_definition_dir, label_files[i])
    num_class = 0
    labels = []
    with open(label_path, 'r') as f:
      data = f.readlines()
      for line in data:
        words = line.split(splitter)
        labels.append(words[0].replace("\r", "").replace("\n", ""))
        num_class = num_class + 1
    print 'number of classes = %d' % num_class
    num_classes.append(num_class)
    class_labels.append(labels)

  if label_type_count == 1:
    num_classes = num_classes[0]
  return num_classes, class_labels
  
# purge the directory and remove all the temporary image downloads
def purge(directory, ext):
  for f in os.listdir(directory):
    if f.endswith(ext):
      os.remove(os.path.join(directory, f))

# A helper functions to normalized the input/output feed data lengths of all the tasks
# for multi-task deep learning.
def multitask_data_length_normalization(input_data_feeds, output_data_feeds, input_group_sizes):
  if len(output_data_feeds) != len(input_group_sizes):
    raise Exception("output_data_feeds and input_group_sizes must have the same length")
    
  max_data_length = max(map(lambda x: len(x), output_data_feeds))
  offset = 0
  for g in range(len(output_data_feeds)):
    if len(output_data_feeds[g]) < max_data_length:
      if isinstance(output_data_feeds[g].array, list):
        rep_count = int(max_data_length / len(output_data_feeds[g].array)) + 1
        output_data_feeds[g].array = output_data_feeds[g].array * rep_count
        output_data_feeds[g].array = output_data_feeds[g].array[0:max_data_length]
      else:
        rep_count = int(max_data_length / output_data_feeds[g].array.shape[0]) + 1
        output_data_feeds[g].array = np.tile(output_data_feeds[g].array, (rep_count, 1))
        output_data_feeds[g].array = output_data_feeds[g].array[0:max_data_length,:]

      for h in range(input_group_sizes[g]):
        input_data_feeds[offset + h].array = input_data_feeds[offset + h].array * rep_count
        input_data_feeds[offset + h].array = input_data_feeds[offset + h].array[0:max_data_length]
    
    print "Task ", g ,": normalized output array dimension: ", len(output_data_feeds[g].array) if isinstance(output_data_feeds[g].array, list) else output_data_feeds[g].array.shape
    offset = offset + input_group_sizes[g]
  return input_data_feeds, output_data_feeds
