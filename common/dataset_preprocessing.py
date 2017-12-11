# ***************
# dataset_preprocessing.py
# ***************
# Helper functions to pre-process the input image based
# on the statistic of datasets
# (Author : Yu Chen)
# ***************

# Necessary Imports
from __future__ import division, absolute_import

import dlib
import os
import numpy as np
import PIL.Image

from skimage import io, transform
from tflearn.data_utils import load_image, resize_image, pil_to_nparray

# Preprocess the garment image based on Cantor dataset standard, which will crop the
# input with certain margins and resize
def cantor_image_preprocessing(input_image, target_width, target_height, cropping, keep_alpha=False):
  
  width, height = input_image.size
  aspect_ratio = float(height) / float(width)
  if input_image.mode == 'RGBA' and not keep_alpha:
    # remove the alpha channel
    output_image = PIL.Image.new("RGB", input_image.size, (255, 255, 255))
    output_image.paste(input_image, mask=input_image.split()[3]) # 3 is the alpha channel
  else:
    output_image = input_image
  if cropping != None:
    if cropping == "normal":
      # crop the image
      lmargin = 0.2
      rmargin = 0.2
      umargin = 0.15
      dmargin = 0.05
    elif cropping == "horizontal": # only crop the left and right margins to maintain the aspect ratio
      lmargin = 0.125
      rmargin = 0.125
      umargin = 0
      dmargin = 0
    elif cropping == "wide":
      lmargin = (1.5 - aspect_ratio) / 2
      rmargin = (1.5 - aspect_ratio) / 2
      umargin = 0
      dmargin = 0
    elif cropping == "abof_original":
      lmargin = 0.1
      rmargin = 0.1
      umargin = 0.15
      dmargin = 0.05
    elif cropping == "abof_shop":
      lmargin = 0.1
      rmargin = 0.1
      umargin = 0.05
      dmargin = 0.05
    elif cropping == "abof_shop_wide":
      lmargin = 0.056
      rmargin = 0.056
      umargin = 0
      dmargin = 0
    elif cropping == "abof_narrowed":
      lmargin = -0.0833
      rmargin = -0.0833
      umargin = 0
      dmargin = 0
    else:     
      lmargin = 0
      rmargin = 0
      umargin = 0
      dmargin = 0
    output_image = output_image.crop((int(lmargin * float(width)),
                                      int(umargin * float(height)),
                                      int(float(width) * (1.0 - rmargin)),
                                      int(float(height) * (1.0 - dmargin))))
              
  # resize the image to the target dimension (i.e. 256x256)
  output_image = resize_image(output_image, target_width, target_height)
  output_image = pil_to_nparray(output_image)
  
  # normalise to 0-1
  output_image /= 255.
  return output_image

# Preprocess the face image based on CelebA dataset standard
# This will perform a face landmark detection (using dlib face detector) and crop 
# the image based on the face detection results
def celebA_image_preprocessing(image_path, target_width, target_height, detector, predictor, cropping):
  
  # Load the image.
  input_image = load_image(image_path)
  # NB: horrible hack for dlib face detection, need to load image twice in different format
  skimg = io.imread(image_path)
  
  width, height = input_image.size
  aspect_ratio = float(height) / float(width)
  if input_image.mode == 'RGBA':
    # remove the alpha channel
    output_image = PIL.Image.new("RGB", input_image.size, (255, 255, 255))
    output_image.paste(input_image, mask=input_image.split()[3]) # 3 is the alpha channel
  else:
    output_image = input_image
  if cropping != None:
    #skimg = pil_to_nparray(output_image)

    # dlib face detection
    dets = detector(skimg)
    print("Number of faces detected: {}".format(len(dets)))
        
    if len(dets) == 0:
      # try rotating the image by 90 degrees
      output_image = output_image.rotate(270, expand = True)
      
      # TODO: unwind this save and load hack
      image_path_rotated = image_path + "_rotated.jpg"
      output_image.save(image_path_rotated)
      skimg = io.imread(image_path_rotated)
      dets = detector(skimg)
      print("Rotated. Number of faces detected: {}".format(len(dets)))
      if len(dets) == 0:
        raise Exception('No face detected!')
    
    # Detect the face landmarks
    shape = predictor(skimg, dets[0])
    cx = (int(shape.part(36).x) + int(shape.part(45).x)) / 2 # mid-point of left and right eye corners
    cy = int(shape.part(33).y)  # nose tip
    
    d = abs((int(shape.part(36).y) + int(shape.part(45).y) - int(shape.part(48).y) - int(shape.part(54).y)) / 2)
    d = min(d, int(cx / 2), int((width - 1 - cx) / 2), int(cy / 2.5), int((height - 1 - cy) / 2.5))
    # crop the image
    bbl = max(0, cx - 2 * d)
    bbr = min(width - 1, cx + 2 * d)
    bbu = max(0, int(cy - 2.5 * d))
    bbd = min(height - 1, int(cy + 2.5 * d))
    #print cx, cy, d, bbl, bbr, bbu, bbd
    output_image = output_image.crop((bbl, bbu, bbr, bbd))
    #output_image.save("cropped.jpg")
              
  # resize the image to the target dimension (i.e. 256x256)
  output_image = resize_image(output_image, target_width, target_height)
  output_image = pil_to_nparray(output_image)
  
  # normalise to 0-1
  output_image /= 255.
  return output_image
  
