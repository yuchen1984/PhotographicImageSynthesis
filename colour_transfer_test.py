import os,numpy as np
from os.path import dirname, exists, join, splitext
import json,scipy
from PIL import Image, ImageOps
import colorsys


def pil_to_nparray(pil_image):
    """ Convert a PIL.Image to numpy array. """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, arr_t, mask):
    r, g, b = (np.rollaxis(arr, axis=-1)) / 255.0
    r2, g2, b2 = np.rollaxis(arr_t, axis=-1) / 255.0

    h, s, v = rgb_to_hsv(r, g, b)
    h2, s2, v2 = rgb_to_hsv(r2, g2, b2)
    ro, go, bo = hsv_to_rgb(h2 * mask + h * (1 - mask), s2 * mask + s * (1 - mask), v)
    arr_out = np.dstack((ro, go, bo)) * 255.0
    return arr_out

def colorize(img, avg_img_t, mask):
    """
    Re-Colorize PIL image `original` with the given
    `hue` of another average colour image; returns another PIL image.
    """
    arr = np.array(np.asarray(img).astype('float'))
    arr_t = np.array(np.asarray(avg_img_t).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, arr_t, mask).astype('uint8'), 'RGB')

    return new_img
        
    
working_dir = './testdata'
image_name = '11182_112822_5'
#image_name = '9821_134993_1'
#image_name = '9817_135005_2'
#image_name = '5_311937_2'
#image_name = '4_126332_1'
modifier = 'm0'
file_name_in = 'a_' + image_name + '.jpg'
file_name_out = 'a_' + image_name + '_' + modifier + '.jpg'
file_name_in_label = 'l_' + image_name + '.png'
file_name_out_avg = 'a_' + image_name + '_' + modifier +'.png'

n_classes=11
label_mask = [6, 7, 8]
    
img = Image.open(os.path.join(working_dir, file_name_in))
label = Image.open(os.path.join(working_dir, file_name_in_label))
avg_img_out = Image.open(os.path.join(working_dir, file_name_out_avg))
img = img.convert('RGB')
avg_img_out = avg_img_out.convert('RGB')

iw, ih = img.size
img = pil_to_nparray(img)
label = pil_to_nparray(label)
avg_img_out = pil_to_nparray(avg_img_out)

mask=np.zeros((iw, ih),dtype=np.float32)
for k in range(len(label_mask)):
  mask[:,:]=np.maximum(mask[:,:], np.float32((label[:,:] == label_mask[k])))

img_out = colorize(img, avg_img_out, mask)
img_out.save(file_name_out)

