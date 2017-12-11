from __future__ import division
import os,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import json
import sys

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.core import input_data
from tflearn.models.dnn import DNN
from tflearn.objectives import categorical_crossentropy
from tflearn.layers.core import fully_connected, flatten, input_data
from tflearn.config import _EPSILON, _FLOATX
from tflearn.utils import get_incoming_shape

# Local dependencies
#sys.path.append('..')
tf.logging.set_verbosity(tf.logging.ERROR)

from common.load_data import load_segmentation_label_data, load_multi_dimensional_data, \
                             load_multi_label_data, load_triplet_similarity_data, \
                             multitask_data_length_normalization
from common.multitask_deep_network_builder import create_dnn, create_multitask_data_inputs


# Load task configurations from the JSON file
def load_tasks(task_config_file_path = None):
  if not os.path.isfile(task_config_file_path):
    raise Exception(task_config_file_path + ' does not exist!')

  with open(task_config_file_path, 'r') as fr:
    tasks = json.load(fr)
  return tasks


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

def build_vgg19(input,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net={}
    vgg_rawnet=scipy.io.loadmat('VGG_Model/imagenet-vgg-verydeep-19.mat')
    vgg_layers=vgg_rawnet['layers'][0]
    net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
    net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
    net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
    net['pool1']=build_net('pool',net['conv1_2'])
    net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
    net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
    net['pool2']=build_net('pool',net['conv2_2'])
    net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
    net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
    net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
    net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
    net['pool3']=build_net('pool',net['conv3_4'])
    net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
    net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
    net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
    net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
    net['pool4']=build_net('pool',net['conv4_4'])
    net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
    net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
    net['conv5_3']=build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32),name='vgg_conv5_3')
    net['conv5_4']=build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34),name='vgg_conv5_4')
    net['pool5']=build_net('pool',net['conv5_4'])
    return net

def recursive_generator(label,input_image,sp):
    dim=512 if sp>=128 else 1024
    if sp==4:
        input=tf.concat(3,[label,input_image])
    else:
        downsampled=tf.image.resize_area(label,(sp//2,sp//2),align_corners=False)
        downsampled_avg=tf.image.resize_area(input_image,(sp//2,sp//2),align_corners=False)
        input=tf.concat(3,[tf.image.resize_bilinear(recursive_generator(downsampled,downsampled_avg,sp//2),(sp,sp),align_corners=True),label,input_image])
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    if sp==256:
        net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
        net=(net+1.0)/2.0*255.0
    return net

def compute_error(real,fake,label):
    return tf.reduce_mean(tf.abs(fake-real))#simple loss


#os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))#select a GPU with maximum available memory
#os.system('rm tmp')
  
sess=tf.Session()
is_training = True
n_classes = 11


tasks = load_tasks("./abof_recolour_disc.json")


sp=256#spatial resolution: 256x256
with tf.variable_scope(tf.get_variable_scope()):
    label=tf.placeholder(tf.float32,[None,sp,sp,n_classes + 1])
    input_image=tf.placeholder(tf.float32,[None,sp,sp,3])
    y_true=tf.placeholder(tf.float32,[None,2])
    weight=tf.placeholder(tf.float32)
    generator=recursive_generator(label,input_image,sp)
    fake_full_image = generator + input_image
    normalized_fake_full_image = (fake_full_image / 255.0 - 0.723029862732) / 0.269584051053
    ## Create deep neural network of different architecture
    disc_output = create_dnn(network_type='googlenet', input_datas=[normalized_fake_full_image], is_training=False, task_configs=tasks,
                             weight_sharing = 'all', data_scope = "recolour_disc")
    vgg_real=build_vgg19(input_image)
    vgg_fake=build_vgg19(fake_full_image,reuse=True)
    p0=compute_error(vgg_real['input'],vgg_fake['input'],label)
    p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'],label)/1.6
    p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'],tf.image.resize_area(label,(sp//2,sp//2)))/2.3
    p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'],tf.image.resize_area(label,(sp//4,sp//4)))/1.8
    p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'],tf.image.resize_area(label,(sp//8,sp//8)))/2.8
    p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'],tf.image.resize_area(label,(sp//16,sp//16)))*10/0.8#weights lambda are collected at 100th epoch
    pred = tf.clip_by_value(disc_output, tf.cast(_EPSILON, dtype=_FLOATX), tf.cast(1.-_EPSILON, dtype=_FLOATX))
    GAN_loss = categorical_crossentropy(y_true, pred) * weight
    
    #GAN_loss = tf.log(pred[1]) * loss_weight
    #print get_incoming_shape(pred), get_incoming_shape(GAN_loss), get_incoming_shape(p5)
    G_loss=p0+p1+p2+p3+p4+p5+GAN_loss
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

checkpoint_name = "result_256p_crn_gan"

ckpt=tf.train.get_checkpoint_state(checkpoint_name)
if ckpt:
  print('loaded '+ckpt.model_checkpoint_path)
  saver.restore(sess,ckpt.model_checkpoint_path)

else:
  disc_checkpoint_name = "disc_models"

  ckpt_disc=tf.train.get_checkpoint_state(disc_checkpoint_name)
  if ckpt_disc:
        saver=tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith('recolour_disc')])
        print('loaded '+ckpt_disc.model_checkpoint_path)
        saver.restore(sess,ckpt_disc.model_checkpoint_path)
saver=tf.train.Saver(max_to_keep=1000)

# Read all existing image files in the folder

dir_root = '/data/metail_garments/'
dir_in_l = 'abof-site-images-graysegmentation-resize256/'
dir_in_im = 'abof-site-images-resize256/'
dir_in_rc = 'abof-site-images-randrecolour-resize256/'
dir_out = 'abof-recolour-disc-x2-256/'
dir_out_full = os.path.join(dir_root, dir_out)

training_file_list = []
training_tlabels = []
training_labels = []
with open(os.path.join(dir_out_full, 'train.txt'), 'r') as f:
  data = f.readlines()
  for line in data:
     words = line.strip('\n').split(' ')
     filename = os.path.join(dir_root, words[0])
     if filename.endswith('.jpg'):
        l = words[0].replace('.jpg', '.png')
        l = l.replace(dir_in_im, dir_in_l)
        labelfilename = os.path.join(dir_root, l)
     train_label = int(words[1])
     training_file_list.append(filename)
     training_labels.append(labelfilename)
     training_tlabels.append(train_label)

val_file_list = []
val_tlabels = []
val_labels = []
with open(os.path.join(dir_out_full, 'val.txt'), 'r') as f:
  data = f.readlines()
  for line in data:
     words = line.strip('\n').split(' ')
     filename = os.path.join(dir_root, words[0])
     if filename.endswith('.jpg'):
        l = words[0].replace('.jpg', '.png')
        l = l.replace(dir_in_im, dir_in_l)
        labelfilename = os.path.join(dir_root, l)
     val_label = int(words[1])
     val_file_list.append(filename)
     val_labels.append(labelfilename)
     val_tlabels.append(val_label)
     
training_count = len(training_file_list)
testing_count = len(val_file_list)
print 'training_count: ', training_count
print 'testing_count: ', testing_count

if is_training:
    tlabel=np.zeros((1,2),dtype=np.float32)
    tlabel[0,1] = 1.0
    #tlabel.reshape((1,)+tlabel.shape)  
    g_loss=np.zeros(training_count,dtype=float)
    images=[None]*training_count
    label_images=[None]*training_count
    for epoch in range(1,61):
        if os.path.isdir(os.path.join(checkpoint_name, "%04d" % epoch)):
            continue
        cnt=0
        for i in np.random.permutation(training_count):
            file_name = training_file_list[i]
            label_file_name = training_labels[i]
            st=time.time()
            cnt+=1
            if label_images[i] is None:
                label_images[i]=helper.get_index_semantic_map(label_file_name, n_classes)#training label
                images[i]=np.expand_dims(np.float32(scipy.misc.imread(file_name)),axis=0)#training image
                #print label_images[i].shape, images[i].shape, tlabel.shape
            _,G_current,l0,l1,l2,l3,l4,l5,GAN_l=sess.run([G_opt,G_loss,p0,p1,p2,p3,p4,p5,GAN_loss],feed_dict={label:np.concatenate((label_images[i],np.expand_dims(1-np.sum(label_images[i],axis=3),axis=3)),axis=3),input_image:images[i],y_true:tlabel,lr: 1e-4,weight: 10.0})#may try lr:min(1e-6*np.power(1.1,epoch-1),1e-4 if epoch>100 else 1e-3) in case lr:1e-4 is not good
            g_loss[i]=G_current
            print("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),np.mean(l0),np.mean(l1),np.mean(l2),np.mean(l3),np.mean(l4),np.mean(l5),np.mean(GAN_l),time.time()-st))
        os.makedirs(os.path.join(checkpoint_name, "%04d" % epoch))
        target=open(os.path.join(checkpoint_name, "%04d/score.txt" % epoch),'w')
        target.write("%f"%np.mean(g_loss[np.where(g_loss)]))
        target.close()
        saver.save(sess, os.path.join(checkpoint_name, "model.ckpt"))
        if epoch%20==0:
            saver.save(sess, os.path.join(checkpoint_name, "model_%04d.ckpt" % epoch))

        for i in range(testing_count):
            file_name = val_file_list[i]
            file_name_base = os.path.basename(file_name)
            label_file_name = val_labels[i]
            semantic=helper.get_index_semantic_map(label_file_name, n_classes)#test label
            test_self_image=np.expand_dims(np.float32(scipy.misc.imread(file_name)),axis=0)#test average image
            output=sess.run(generator,feed_dict={label:np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3),input_image:test_self_image})
            full_image = output[0,:,:,:] + test_self_image[0,:,:,:]
            output=np.minimum(np.maximum(output,0.0),255.0)
            full_image=np.minimum(np.maximum(full_image,0.0),255.0)
            #scipy.misc.toimage(output[0,:,:,:],cmin=0,cmax=255).save(os.path.join(checkpoint_name, "%04d/%s_diff_output.png"%(epoch,file_name_base)))
            scipy.misc.toimage(full_image,cmin=0,cmax=255).save(os.path.join(checkpoint_name, "%04d/%s"%(epoch,file_name_base)))


if not os.path.isdir(os.path.join(checkpoint_name, "final")):
    os.makedirs(os.path.join(checkpoint_name, "final"))

  
for i in range(testing_count):
    file_name = val_file_list[i]
    file_name_base = os.path.basename(file_name)
    label_file_name = val_labels[i]
    semantic=helper.get_index_semantic_map(label_file_name, n_classes)#test label
    test_self_image=np.expand_dims(np.float32(scipy.misc.imread(file_name)),axis=0)#test average image
    output=sess.run(generator,feed_dict={label:np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3),input_image:test_self_image})
    full_image = output[0,:,:,:] + test_self_image[0,:,:,:]
    output=np.minimum(np.maximum(output, 0.0), 255.0)
    full_image=np.minimum(np.maximum(full_image,0.0),255.0)
    #scipy.misc.toimage(output[0,:,:,:],cmin=0,cmax=255).save(os.path.join(checkpoint_name, "final/%s_diff_output.png"%(file_name_base)))
    scipy.misc.toimage(full_image,cmin=0,cmax=255).save(os.path.join(checkpoint_name, "final/%s"%(file_name_base)))

    
