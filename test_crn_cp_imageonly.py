from __future__ import division
import argparse
import os,time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import traceback
from PIL import Image

def pil_to_nparray(pil_image):
    """ Convert a PIL.Image to numpy array. """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

def lrelu(x):
    return tf.maximum(0.2*x,x)

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

def recursive_generator(noshadow_image,sp):
    dim=512 if sp>=128 else 1024
    if sp==512:
        dim=128
    if sp==1024:
        dim=32
    if sp==4:
        input=noshadow_image
    else:
        downsampled_noshadow=tf.image.resize_bilinear(noshadow_image,(sp//2,sp//2),align_corners=False)
        input=tf.concat(3,[tf.image.resize_bilinear(recursive_generator(downsampled_noshadow,sp//2),(sp,sp),align_corners=True),noshadow_image])
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    if sp==1024:
        net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
        net=(net+1.0)/2.0*255.0
    return net

# Basic model parameters as external flags.
parser = argparse.ArgumentParser()
parser.add_argument("-wd", "--working_dir", type=str, help="Working directory", default="./")
parser.add_argument("-a", "--noshadow_image", type=str, help="No shadow image name", default="no_shadow.png")
parser.add_argument("-o", "--output_image", type=str, help="Output image name", default="output.png")
parser.add_argument("-cn", "--checkpoint_name", type=str, help="Checkpoint name", default="result_1024p_cp_imonly")
parser.add_argument("-sp", "--resolution", type=int, help="Image height", default=1024)
args = parser.parse_args()

try:

  #os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
  #os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))#select a GPU with maximum available memory
  #os.system('rm tmp')
  sess=tf.Session()
  sp=args.resolution 

  with tf.variable_scope(tf.get_variable_scope()):
      noshadow_image=tf.placeholder(tf.float32,[None,None,None,3])
      generator=recursive_generator(noshadow_image,sp)
      reconstruction = noshadow_image - generator
      reconstruction = tf.clip_by_value(reconstruction, tf.cast(0.0, dtype=tf.float32), tf.cast(255.0, dtype=tf.float32))
  sess.run(tf.global_variables_initializer())

  checkpoint_name = args.checkpoint_name

  ckpt=tf.train.get_checkpoint_state(checkpoint_name)
  if ckpt:
      print('loaded '+ckpt.model_checkpoint_path)
      saver=tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
      saver.restore(sess,ckpt.model_checkpoint_path)
  else:
      raise Exception("Invalid check point")

  if not os.path.isfile(os.path.join(args.working_dir, args.noshadow_image)):#test average image not exist
      raise Exception("Invalid label image or average image path")
      
  st=time.time()    
  pil_noshadow_image = Image.open(os.path.join(args.working_dir, args.noshadow_image))
  iw, ih = pil_noshadow_image.size
  noshadow_image_arr = pil_to_nparray(pil_noshadow_image)
  if iw != sp or ih != sp:
    pil_noshadow_image = pil_noshadow_image.resize((sp, sp), Image.ANTIALIAS)
  test_noshadow_image=np.expand_dims(pil_to_nparray(pil_noshadow_image),axis=0)
  #pil_test_noshadow_image = Image.fromarray(np.uint8(test_noshadow_image[0,:,:,:]),mode='RGB')
  #pil_test_noshadow_image.info = pil_noshadow_image.info
  #pil_test_noshadow_image.save(os.path.join(args.working_dir, "input_" + args.output_image))

  st0=time.time()    
  output=sess.run(reconstruction,feed_dict={noshadow_image:test_noshadow_image})
  print("inference done:  %.2f"%(time.time()-st0))

  diff_image = test_noshadow_image[0,:,:,:] - output[0,:,:,:]
  output=np.minimum(np.maximum(output, 0.0), 255.0)
  diff_image=np.minimum(np.maximum(diff_image,0.0),255.0)
  pil_diff_image = Image.fromarray(np.uint8(diff_image),mode='RGB')
  pil_diff_image.info = pil_noshadow_image.info
  pil_diff_image.save(os.path.join(args.working_dir, "diff_" + args.output_image))
  pil_full_image = Image.fromarray(np.uint8(output[0,:,:,:]),mode='RGB')
  pil_full_image.info = pil_noshadow_image.info
  pil_full_image.save(os.path.join(args.working_dir, args.output_image))
  if iw != sp or ih != sp:
    pil_diff_image = pil_diff_image.resize((iw, ih), Image.ANTIALIAS)
    #pil_diff_image.save(os.path.join(args.working_dir, "diff_fullsize_" + args.output_image))
    allshadow_image_arr = noshadow_image_arr - pil_to_nparray(pil_diff_image)
    allshadow_image_arr=np.minimum(np.maximum(allshadow_image_arr, 0.0), 255.0)
    pil_full_image_fullsize = Image.fromarray(np.uint8(allshadow_image_arr),mode='RGB')
    pil_full_image_fullsize.info = pil_noshadow_image.info
    pil_full_image_fullsize.save(os.path.join(args.working_dir, "fullsize_" + args.output_image))    
  print("done:  %.2f"%(time.time()-st))
except Exception as err:
  print('Quit by errors: ' , err)
  traceback.print_exc()

