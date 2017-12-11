# ***************
# image_utils.py
# ***************
# Helper functions for image encoding, decoding and transformation
# (Author : Yu Chen)
# ***************

# ********************************************************
# Necessary Imports
from PIL import Image
import numpy as np
               
# ------------------------------------------------------
def decode_colour_labels(mask, label_color_map):
    """Decode batch of segmentation masks.
    
    Args:
      label_batch: result of inference after taking argmax.
    
    Returns:
      An batch of RGB images of the same size
    """
    img = Image.new('RGB', (len(mask[0]), len(mask)))
    num_classes = len(label_color_map)
    pixels = img.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_,j_] = label_color_map[k]
    return np.array(img)
    

