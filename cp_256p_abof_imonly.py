from __future__ import division
import os,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from PIL import Image
from tflearn.config import _FLOATX
from tflearn.data_utils import load_image, pil_to_nparray

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

def recursive_generator(avg_image,sp):
    dim=512 if sp>=128 else 1024
    if sp==4:
        input=tf.concat(3,[avg_image])
    else:
        downsampled_avg=tf.image.resize_bilinear(avg_image,(sp//2,sp//2),align_corners=False)
        input=tf.concat(3,[tf.image.resize_bilinear(recursive_generator(downsampled_avg,sp//2),(sp,sp),align_corners=True),avg_image])
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    if sp==256:
        net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
        net=(net+1.0)/2.0*255.0
    return net

def compute_error(real,fake):
    return tf.reduce_mean(tf.abs(fake-real))#simple loss

#os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))#select a GPU with maximum available memory
#os.system('rm tmp')
sess=tf.Session()
is_training = True
sp=256#spatial resolution: 256x256
with tf.variable_scope(tf.get_variable_scope()):
    real_image=tf.placeholder(tf.float32,[None,None,None,3])
    avg_image=tf.placeholder(tf.float32,[None,None,None,3])
    generator=recursive_generator(avg_image,sp)
    fake_full_image = avg_image - generator
    fake_full_image = tf.clip_by_value(fake_full_image, tf.cast(0.0, dtype=_FLOATX), tf.cast(255.0, dtype=_FLOATX))
    weight=tf.placeholder(tf.float32)
    vgg_real=build_vgg19(real_image)
    vgg_fake=build_vgg19(fake_full_image,reuse=True)
    p0=compute_error(vgg_real['input'],vgg_fake['input'])
    p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'])/1.6
    p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'])/2.3
    p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'])/1.8
    p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'])/2.8
    p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'])*10/0.8#weights lambda are collected at 100th epoch
    G_loss=p0+p1+p2+p3+p4+p5
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

checkpoint_name = "result_256p_cp_imonly"

ckpt=tf.train.get_checkpoint_state(checkpoint_name)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

# Read all existing image files in the folder
dir_image = "/data/metail_garments/cp-as-resize256/"
#dir_diff_image = "/data/metail_garments/cp-diff-resize256/"
dir_self_image = "/data/metail_garments/cp-ns-resize256/"
file_list = []
for f in os.listdir(dir_self_image):
  if f.endswith(".png"):
    #print(f)
    file_list.append(f)
training_count = int(len(file_list) * 0.9)
testing_count = len(file_list) - training_count
print 'training_count: ', training_count
print 'testing_count: ', testing_count

if is_training:
    g_loss=np.zeros(training_count,dtype=float)
    images=[None]*training_count
    self_images=[None]*training_count
    for epoch in range(1,201):
        if os.path.isdir(os.path.join(checkpoint_name, "%04d" % epoch)):
            continue
        cnt=0
        for i in np.random.permutation(training_count):
            file_name = file_list[i]
            st=time.time()
            cnt+=1
            if images[i] is None:
                self_images[i]=np.expand_dims(pil_to_nparray(load_image(os.path.join(dir_self_image, file_name))),axis=0)#training average image
                images[i]=np.expand_dims(pil_to_nparray(load_image(os.path.join(dir_image, file_name.replace('_ns','_as')))),axis=0)#training image
            _,G_current,l0,l1,l2,l3,l4,l5=sess.run([G_opt,G_loss,p0,p1,p2,p3,p4,p5],feed_dict={real_image:images[i],avg_image:self_images[i],lr: 1e-4})#may try lr:min(1e-6*np.power(1.1,epoch-1),1e-4 if epoch>100 else 1e-3) in case lr:1e-4 is not good
            g_loss[i]=G_current
            print("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),np.mean(l0),np.mean(l1),np.mean(l2),np.mean(l3),np.mean(l4),np.mean(l5),time.time()-st))
        os.makedirs(os.path.join(checkpoint_name, "%04d" % epoch))
        target=open(os.path.join(checkpoint_name, "%04d/score.txt" % epoch),'w')
        target.write("%f"%np.mean(g_loss[np.where(g_loss)]))
        target.close()
        saver.save(sess, os.path.join(checkpoint_name, "model.ckpt"))
        if epoch%20==0:
            saver.save(sess, os.path.join(checkpoint_name, "model_%04d.ckpt" % epoch))

        for i in range(testing_count):
            file_name = file_list[training_count + i]

            if not os.path.isfile(os.path.join(dir_self_image, file_name)):#test average image
                continue
            timg = load_image(os.path.join(dir_self_image, file_name))
            test_self_image=np.expand_dims(pil_to_nparray(timg),axis=0)#test average image
            pil_test_self_image = Image.fromarray(np.uint8(test_self_image[0,:,:,:]),mode='RGB')
            pil_test_self_image.info = timg.info
            pil_test_self_image.save(os.path.join(checkpoint_name, "%04d/%s_input.png"%(epoch,file_name)))
            output=sess.run(generator,feed_dict={avg_image:test_self_image})
            full_image =  test_self_image[0,:,:,:] - output[0,:,:,:]
            output=np.minimum(np.maximum(output,0.0),255.0)
            full_image=np.minimum(np.maximum(full_image,0.0),255.0)
            pil_diff_image = Image.fromarray(np.uint8(output[0,:,:,:]),mode='RGB')
            pil_diff_image.info = timg.info
            pil_diff_image.save(os.path.join(checkpoint_name, "%04d/%s_diff_output.png"%(epoch,file_name)))
            pil_full_image = Image.fromarray(np.uint8(full_image),mode='RGB')
            pil_full_image.info = timg.info
            pil_full_image.save(os.path.join(checkpoint_name, "%04d/%s_output.png"%(epoch,file_name)))


if not os.path.isdir(os.path.join(checkpoint_name, "final")):
    os.makedirs(os.path.join(checkpoint_name, "final"))
    
for i in range(testing_count):
    file_name = file_list[training_count + i]
    if not os.path.isfile(os.path.join(dir_self_image, file_name)):#test average image
        continue
    timg = load_image(os.path.join(dir_self_image, file_name))
    test_self_image=np.expand_dims(pil_to_nparray(timg),axis=0)#test average image
    pil_test_self_image = Image.fromarray(np.uint8(test_self_image[0,:,:,:]),mode='RGB')
    pil_test_self_image.info = timg.info
    pil_test_self_image.save(os.path.join(checkpoint_name, "final/%s_input.png"%(file_name)))
    output=sess.run(generator,feed_dict={avg_image:test_self_image})
    full_image = test_self_image[0,:,:,:] - output[0,:,:,:]
    output=np.minimum(np.maximum(output, 0.0), 255.0)
    full_image=np.minimum(np.maximum(full_image,0.0),255.0)
    pil_diff_image = Image.fromarray(np.uint8(output[0,:,:,:]),mode='RGB')
    pil_diff_image.info = timg.info
    pil_diff_image.save(os.path.join(checkpoint_name, "final/%s_diff_output.png"%(file_name)))
    pil_full_image = Image.fromarray(np.uint8(full_image),mode='RGB')
    pil_full_image.info = timg.info
    pil_full_image.save(os.path.join(checkpoint_name, "final/%s_output.png"%(file_name)))
    
