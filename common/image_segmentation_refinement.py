import cv2
import numpy as np
import os
import os.path
from PIL import Image

def refine_segmentation(img, label_image, label_mapping_body, label_mapping_garment, target_width, target_height, run_grabcut = 1):
  img_resize = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR) 
  margin = target_height / 150;
  label_image_resize = cv2.resize(label_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST) 
  
  # Simply Resize the original segmentation
  if label_mapping_garment == None or label_mapping_body == None:
    return label_image_resize
    
  mask_garment = np.zeros(img_resize.shape[:2],np.uint8)
  for pair in label_mapping_garment:
    mask_garment[label_image_resize == pair[0]] = 255 if pair[1] > 0 else 0
  
  # Compute connnected components
  cc_output = cv2.connectedComponentsWithStats(mask_garment, 8, cv2.CV_32S)
  labels = cc_output[1]
  stats = cc_output[2]
  
  # Pick the largest connected component as the initialisation
  largest_area_index = np.argsort(-stats[:,-1])[1]
  largest_cc = np.zeros(img_resize.shape[:2],np.uint8)
  largest_cc[labels == largest_area_index] = 255

  # erode the largest connected component for the sure foreground
  kernel = np.ones((margin,margin),np.uint8)
  largest_cc_erode = cv2.erode(largest_cc, kernel, iterations=1)
  # get the bounding box of the largest connected component
  bbx_min = max(0, stats[1,0] - margin)
  bby_min = max(0, stats[1,1] - margin)
  bbx_max = min(target_width, stats[1,0] + stats[1,2] + margin)
  bby_max = min(target_height, stats[1,1] + stats[1,3] + margin)
  
  if run_grabcut >= 2:  # Two stage GrabCut
    # Specify the basic model 
    mask = np.zeros(img_resize.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # Specify the rectangle which definitely contains the foreground 
    rect = (1,1,target_width,target_height)
    
    # 0 = Sure Background, 1 = Sure Foreground, 2 = Prob Background, 3 = Prob Foreground 
    for pair in label_mapping_body:
      mask[label_image_resize == pair[0]] = 2 + pair[1]
    
    # Do GrabCut - Stage 1:  body vs. background
    cv2.grabCut(img_resize,mask,rect,bgdModel,fgdModel,run_grabcut,cv2.GC_INIT_WITH_MASK)

    mask_output1 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_segmentation1 = img_resize * mask_output1[:,:,np.newaxis]
    
    mask = np.zeros(img_resize.shape[:2],np.uint8)
    #bgdModel = np.zeros((1,65),np.float64)
    #fgdModel = np.zeros((1,65),np.float64)
    mask[bbx_min:bbx_max, bby_min:bby_max] = 2
    mask[largest_cc > 0] = 3
    mask[largest_cc_erode > 0] = 1
    mask = mask * mask_output1
    
    # Do GrabCut - Stage 2:  garment vs. rest of the body
    cv2.grabCut(img_segmentation1,mask,rect,bgdModel,fgdModel,run_grabcut,cv2.GC_INIT_WITH_MASK)
    mask_output_final = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_segmentation_final = img_segmentation1 * mask_output_final[:,:,np.newaxis]
    
  elif run_grabcut == 1: # One stage GrabCut
    # Specify the rectangle which definitely contains the foreground 
    rect = (bbx_min,bby_min,bbx_max - bbx_min,bby_max - bby_min)
    
    # Specify the basic colour model 
    mask = np.zeros(img_resize.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # 0 = Sure Background, 1 = Sure Foreground, 2 = Prob Background, 3 = Prob Foreground 
    mask[bbx_min:bbx_max, bby_min:bby_max] = 2
    mask[largest_cc > 0] = 3
    mask[largest_cc_erode > 0] = 1
    
    # Do GrabCut: garment vs. background
    cv2.grabCut(img_resize,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    #cv2.grabCut(img_resize,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK + cv2.GC_INIT_WITH_RECT)
    mask_output_final = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_segmentation_final = img_resize * mask_output_final[:,:,np.newaxis]
  else:  # Resize + Selecting largest connect component
    mask_output_final = np.zeros(img_resize.shape[:2],np.uint8)
    mask_output_final[largest_cc > 0] = 1
    
  return mask_output_final

def generate_foreground_background_masks(probability_image, label_mapping, sure_threshold = 0.99, maybe_threshold = 0.5):
  prob_fg = np.zeros(probability_image.shape[:2],np.float32)
  prob_bg = np.zeros(probability_image.shape[:2],np.float32)
  for pair in label_mapping:
    if pair[1] > 0:
      prob_fg = prob_fg + probability_image[:,:,pair[0]]
    else:
      prob_bg = prob_bg + probability_image[:,:,pair[0]]
  prob_all = prob_fg + prob_bg
  prob_fg = prob_fg / prob_all
  prob_bg = prob_bg / prob_all

  bg_sure = prob_bg > sure_threshold
  bg_unsure = np.logical_and(prob_bg <= sure_threshold, prob_bg > 1.0 - maybe_threshold)
  fg_sure = prob_fg > sure_threshold
  fg_unsure = np.logical_and(prob_fg <= sure_threshold, prob_fg >= maybe_threshold)

  # 0 = Sure Background, 1 = Sure Foreground, 2 = Prob Background, 3 = Prob Foreground 
  mask = np.zeros(probability_image.shape[:2],np.uint8)
  mask[bg_sure] = 0
  mask[fg_sure] = 1
  mask[bg_unsure] = 2
  mask[fg_unsure] = 3
  
  # Save the mask image for the debugging purpose
  #tmp = np.zeros(probability_image.shape[:2],np.uint8)
  #tmp[bg_unsure] = 128
  #tmp[fg_unsure] = 192
  #tmp[bg_sure] = 64
  #tmp[fg_sure] = 255
  #cv2.imwrite("mask.png", tmp)
  
  return mask, prob_fg, prob_bg

"""
GrabCut segmentation refinement based on thresholding the probability output, this scheme 
generally gives a lot better boundary compared with GrabCut initialized on the label output. 
"""
  
def refine_segmentation_probability(img, probability_image, label_mapping_body, label_mapping_garment, \
                                    target_width, target_height, run_grabcut = 1, mask_config = None, \
                                    enable_roi = True, roi_margin = 0.02, fg_threshold = 0.5):
  img_resize = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
  # This equivalenty do a Softmax (without normalization)
  max_prob = np.max(probability_image, axis=2)
  probability_image = probability_image - max_prob[:,:,np.newaxis]
  probability_image = np.exp(probability_image)
  probability_image_resize = cv2.resize(probability_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR) 
  #probability_image_resize = np.exp(probability_image_resize)
  
  # Simply Resize the original segmentation
  if label_mapping_garment == None or label_mapping_body == None:
    label_image_resize = np.argmax(probability_image_resize, axis=2)
    return label_image_resize
    
  rect = (1,1,target_width,target_height)
  mask_garment, prob_fg_garment, prob_bg_garment = generate_foreground_background_masks(probability_image_resize, label_mapping_garment, \
                                                                                        maybe_threshold = fg_threshold)

  if run_grabcut >= 1:
    mask_cc = np.zeros(img_resize.shape[:2],np.uint8)
    mask_cc[prob_fg_garment >= fg_threshold] = 1
    
    # Compute connnected components
    cc_output = cv2.connectedComponentsWithStats(mask_cc, 8, cv2.CV_32S)
    labels = cc_output[1]
    stats = cc_output[2]
    
    # Pick the largest connected component as the initialisation
    largest_area_index = np.argsort(-stats[:,-1])[1]

    # get the bounding box of the largest connected component
    bbx_min = max(0, stats[1,0])
    bby_min = max(0, stats[1,1])
    bbx_max = min(target_width, stats[1,0] + stats[1,2])
    bby_max = min(target_height, stats[1,1] + stats[1,3])
    mask_ex = np.zeros(img_resize.shape[:2],np.uint8)
    margin = int(target_height * roi_margin);
    roix_min = max(0, stats[1,0] - margin)
    roiy_min = max(0, stats[1,1] - margin)
    roix_max = min(target_width, stats[1,0] + stats[1,2] + margin)
    roiy_max = min(target_height, stats[1,1] + stats[1,3] + margin)
    
    if mask_config != None:
      reference_y = bby_min
      if mask_config["reference"] == "left":
        reference_x = bbx_min
      elif mask_config["reference"] == "right":
        reference_x = bbx_max
      else:
        reference_x = (bbx_min + bbx_max) / 2
      
      if mask_config["mode"] == "left":
        x_min = max(0, reference_x - target_width)
        y_min = max(0, reference_y + int(target_height * mask_config["yoffset_min"]))
        x_max = min(target_width, reference_x + int(target_width * mask_config["xoffset_min"]))
        y_max = min(target_height, reference_y + int(target_height * mask_config["yoffset_max"]))
        mask_ex[y_min:y_max, x_min:x_max] = 1
      elif mask_config["mode"] == "right":
        x_min = max(0, reference_x - int(target_width * mask_config["xoffset_min"]))
        y_min = max(0, reference_y + int(target_height * mask_config["yoffset_min"]))
        x_max = min(target_width, reference_x + target_width)
        y_max = min(target_height, reference_y + int(target_height * mask_config["yoffset_max"]))
        mask_ex[y_min:y_max, x_min:x_max] = 1
      else:
        x_min1 = max(0, reference_x - target_width)
        x_min2 = max(0, reference_x + int(target_width * mask_config["xoffset_min"]))
        y_min = max(0, reference_y + int(target_height * mask_config["yoffset_min"]))
        x_max1 = min(target_width, reference_x - int(target_width * mask_config["xoffset_min"]))
        x_max2 = min(target_width, reference_x + target_width)
        y_max = min(target_height, reference_y + int(target_height * mask_config["yoffset_max"]))
        mask_ex[y_min:y_max, x_min1:x_max1] = 1
        mask_ex[y_min:y_max, x_min2:x_max2] = 1
      
      #cv2.imwrite("maskex.png", mask_ex * 255) 
  
  if run_grabcut >= 2 and len(label_mapping_garment) > 2:  # Two stage GrabCut
    # Specify the basic model 
    mask1, prob_fg_body, prob_bg_body = generate_foreground_background_masks(probability_image_resize, label_mapping_body)
    
    # Do GrabCut - Stage 1:  body vs. background
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img_resize,mask1,rect,bgdModel,fgdModel,run_grabcut,cv2.GC_INIT_WITH_MASK)

    mask_output1 = np.where((mask1==2)|(mask1==0),0,1).astype('uint8')
    img_segmentation1 = img_resize * mask_output1[:,:,np.newaxis]
    mask2 = mask_garment * mask_output1
    
    # Do GrabCut - Stage 2:  garment vs. rest of the body
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img_segmentation1,mask2,rect,bgdModel,fgdModel,run_grabcut,cv2.GC_INIT_WITH_MASK)
    mask_output_final = np.where((mask2==2)|(mask2==0),0,1).astype('uint8')
    img_segmentation_final = img_segmentation1 * mask_output_final[:,:,np.newaxis]
    
  elif run_grabcut >= 1: # One stage GrabCut
    # Specify the basic colour model 
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask = mask_garment
    img_input = img_resize
    if mask_config != None:
      #mask = mask_garment * (1 - mask_ex) + mask_cc * mask_ex
      mask_c3 = np.where(((1 - mask_cc) *  mask_ex == 1),1,0).astype('uint8')
      
      img_input = img_resize * (1 - mask_c3[:,:,np.newaxis]) + 255 * mask_c3[:,:,np.newaxis]
      
    # Clean up the content outside the roi. This helps a lot for the GrabCut cutout
    if enable_roi:
      mask_roi = np.zeros(img_resize.shape[:2],np.uint8)
      mask_roi[roiy_min:roiy_max, roix_min:roix_max] = 1
      img_input = img_input * mask_roi[:,:,np.newaxis] + 255 * (1 - mask_roi[:,:,np.newaxis])
      #cv2.imwrite("img_input.jpg", img_input)
      
    # Do GrabCut:  garment vs. background
    cv2.grabCut(img_input,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_MASK)
    mask_output_final = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_segmentation_final = img_resize * mask_output_final[:,:,np.newaxis]
  else:  # Resize + Selecting largest connect component
    mask_output_final = np.zeros(img_resize.shape[:2],np.uint8)
    mask_output_final[prob_fg_garment >= fg_threshold] = 1
    
  return mask_output_final  
