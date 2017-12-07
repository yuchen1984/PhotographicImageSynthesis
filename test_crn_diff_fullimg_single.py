from __future__ import division
import argparse
import os,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import traceback

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

def recursive_generator(label,avg_image,sp):
    dim=512 if sp>=128 else 1024
    if sp==512:
        dim=128
    if sp==1024:
        dim=32
    if sp==4:
        input=tf.concat(3,[label,avg_image])
    else:
        downsampled=tf.image.resize_area(label,(sp//2,sp//2),align_corners=False)
        downsampled_avg=tf.image.resize_area(avg_image,(sp//2,sp//2),align_corners=False)
        input=tf.concat(3,[tf.image.resize_bilinear(recursive_generator(downsampled,downsampled_avg,sp//2),(sp,sp),align_corners=True),label,avg_image])
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    if sp==1024:
        net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
        net=(net+1.0)/2.0*255.0
    return net

# Basic model parameters as external flags.
parser = argparse.ArgumentParser()
parser.add_argument("-wd", "--working_dir", type=str, help="Working directory", default="./testdata/")
parser.add_argument("-l", "--label_image", type=str, help="Label image name", default="label.png")
parser.add_argument("-a", "--average_image", type=str, help="Average image name", default="average.jpg")
parser.add_argument("-o", "--output_image", type=str, help="Output image name", default="output.jpg")
parser.add_argument("-cn", "--checkpoint_name", type=str, help="Checkpoint name", default="result_1024p_abof_diff_fullimg_ft")
parser.add_argument("-sp", "--resolution", type=int, help="Image height", default=1024)
parser.add_argument("-nc", "--num_classes", type=int, help="Num of segmentation label classes", default=11)
args = parser.parse_args()

try:

  #os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
  #os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))#select a GPU with maximum available memory
  #os.system('rm tmp')
  sess=tf.Session()
  n_classes = args.num_classes
  sp=args.resolution 

  with tf.variable_scope(tf.get_variable_scope()):
      label=tf.placeholder(tf.float32,[None,None,None,n_classes + 1])
      avg_image=tf.placeholder(tf.float32,[None,None,None,3])
      generator=recursive_generator(label,avg_image,sp)
      reconstruction = generator + avg_image - 128
  sess.run(tf.global_variables_initializer())

  checkpoint_name = args.checkpoint_name

  ckpt=tf.train.get_checkpoint_state(checkpoint_name)
  if ckpt:
      print('loaded '+ckpt.model_checkpoint_path)
      saver=tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
      saver.restore(sess,ckpt.model_checkpoint_path)
  else:
      raise Exception("Invalid check point")

  if not os.path.isfile(os.path.join(args.working_dir, args.label_image)) or not os.path.isfile(os.path.join(args.working_dir, args.average_image)):#test average image not exist
      raise Exception("Invalid label image or average image path")
      
  semantic=helper.get_index_semantic_map(os.path.join(args.working_dir, args.label_image), n_classes)#test label
  test_avg_image=np.expand_dims(np.float32(scipy.misc.imread(os.path.join(args.working_dir, args.average_image))),axis=0)#test average image
  output=sess.run(reconstruction,feed_dict={label:np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3),avg_image:test_avg_image})
  output=np.minimum(np.maximum(output, 0.0), 255.0)
  scipy.misc.toimage(output[0,:,:,:],cmin=0,cmax=255).save(os.path.join(args.working_dir, args.output_image))
except Exception as err:
  print('Quit by errors: ' , err)
  traceback.print_exc()

