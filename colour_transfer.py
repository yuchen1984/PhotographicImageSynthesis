import os,numpy as np
from os.path import dirname, exists, join, splitext
import json,scipy
from PIL import Image, ImageOps
import argparse
import colorsys
import traceback

def pil_to_nparray(pil_image):
    """ Convert a PIL.Image to numpy array. """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

EPSILON = 1e-9

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
rgb_to_hls = np.vectorize(colorsys.rgb_to_hls)
hls_to_rgb = np.vectorize(colorsys.hls_to_rgb)
rgb_to_yiq = np.vectorize(colorsys.rgb_to_yiq)
yiq_to_rgb = np.vectorize(colorsys.yiq_to_rgb)

def gamma_estimation(source, target, mask):
  sum_intensity_src = np.sum(source * mask)
  sum_intensity_tar = np.sum(target * mask)
  sum_mask = np.sum(mask) + EPSILON
  avg_v_src = sum_intensity_src / sum_mask + EPSILON
  avg_v_tar = sum_intensity_tar / sum_mask + EPSILON
  gamma = np.log(avg_v_tar) / np.log(avg_v_src)
  return gamma

def shift_hue(arr_src, arr_tar, mask, gamma_correction = False, colour_space='hsv'):
    r, g, b = (np.rollaxis(arr_src, axis=-1)) / 255.0
    rt, gt, bt = np.rollaxis(arr_tar, axis=-1) / 255.0
    if colour_space.lower() == 'hls':
      h, l, s = rgb_to_hls(r, g, b)
      ht, lt, st = rgb_to_hls(rt, gt, bt)
      ro, go, bo = hls_to_rgb(ht * mask + h * (1 - mask), l, st * mask + s * (1 - mask))
    else:
      h, s, v = rgb_to_hsv(r, g, b)
      ht, st, vt = rgb_to_hsv(rt, gt, bt)
      ro, go, bo = hsv_to_rgb(ht * mask + h * (1 - mask), st * mask + s * (1 - mask), v)
    
    if gamma_correction:
      # Apply gamma correction to balance the brightness
      yo, io, qo = rgb_to_yiq(ro, go, bo)
      yt, it, qt = rgb_to_yiq(rt, gt, bt)
      gamma = gamma_estimation(yo, yt, mask)
      print 'gamma = ', gamma
      
      ro, go, bo = yiq_to_rgb(np.power(yo, gamma) * mask + yo * (1 - mask), io, qo)
      
    arr_out = np.dstack((ro, go, bo)) * 255.0
    return arr_out

def colorize(src_img, avg_img_target, mask, gamma_correction = False, colour_space='hsv'):
    """
    Re-Colorize PIL image `original` with the given
    `hue` of another average colour image; returns another PIL image.
    """
    arr_src = np.array(np.asarray(src_img).astype('float'))
    arr_tar = np.array(np.asarray(avg_img_target).astype('float'))
    arr_sh = shift_hue(arr_src, arr_tar, mask, gamma_correction, colour_space).astype('uint8')
    new_img = Image.fromarray(arr_sh, 'RGB')
    return new_img
        
# Basic model parameters as external flags.
parser = argparse.ArgumentParser()
parser.add_argument("-wd", "--working_dir", type=str, help="Working directory", default="./testdata/")
parser.add_argument("-s", "--source_image", type=str, help="Source image to be recoloured", default="image.jpg")
parser.add_argument("-a", "--average_target_image", type=str, help="Target colour image with FG mask", default="target.png")
parser.add_argument("-o", "--output_image", type=str, help="Output image name", default="output.png")
parser.add_argument("-g", "--gamma_correction", type=bool, help="If apply gamma correction", default=False)
parser.add_argument("-c", "--colour_space", type=str, help="Colour space to be transformed into", default='hsv')
args = parser.parse_args()

img = Image.open(os.path.join(args.working_dir, args.source_image))
avg_img_out = Image.open(os.path.join(args.working_dir, args.average_target_image))
img = img.convert('RGB')
avg_img_target = avg_img_out.convert('RGB')

iw, ih = img.size
iwt, iht = avg_img_target.size
if iwt != iw or iht != ih:
  avg_img_target = avg_img_target.resize((iw, ih), Image.NEAREST)
  
img = pil_to_nparray(img)
avg_img_target = pil_to_nparray(avg_img_target)

mask=np.zeros((iw, ih),dtype=np.float32)
mask[:,:]=1.0 - np.float32(np.bitwise_and(np.bitwise_and((avg_img_target[:,:,0] == 0), (avg_img_target[:,:,1] == 0)) , (avg_img_target[:,:,2] == 0)))

img_out = colorize(img, avg_img_target, mask, args.gamma_correction, args.colour_space)
img_out.save(os.path.join(args.working_dir, args.output_image))

