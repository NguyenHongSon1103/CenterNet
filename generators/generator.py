import json
import os
import cv2
import tensorflow as tf
import math
import numpy as np
import sys
sys.path.append('/data2/sonnh/E2EObjectDetection/Centernet')
from generators.utils import gaussian_radius, draw_gaussian, draw_gaussian_2, draw_msra_gaussian
from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect

def resize_keep_ar(size, image, boxes):
    h, w, c = image.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)
    padimg = np.zeros((size, size, c), image.dtype)
    padimg[:h, :w] = cv2.resize(image, (w, h))
    new_anns = []
    for box in boxes:
        box = np.array(box).astype(np.float32)
        box *= scale
        new_anns.append(box)
    return padimg, new_anns
    
def resize_wo_keep_ar(size, image, boxes):
    h, w, c = image.shape
    resized = cv2.resize(image, (size, size))
    scale_h, scale_w = size/h, size/w
    new_anns = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xmin, xmax = xmin * scale_w, xmax * scale_w
        ymin, ymax = ymin * scale_h, ymax * scale_h
        new_anns.append([xmin, ymin, xmax, ymax])
    return resized, new_anns
    
resize_methods = {
    'keep':resize_keep_ar,
    'no_keep':resize_wo_keep_ar
}

class Generator(tf.keras.utils.Sequence):
    def __init__(self, data, hparams, mode='train'):
        """
        Initialize Generator object.

        Args:
            data: dictionary with 2 keys: 'im_path' and 'lb_path'
            hparams: a config dictionary
        """
        self.data = data
        self.resizer = resize_methods['keep']
        self.batch_size = hparams['batch_size']
        self.input_size = hparams['input_size']
        self.stride = 4
        self.output_size = self.input_size // self.stride
        self.max_objects = hparams['max_objects']
        self.num_classes = hparams['num_classes']
        self.mode = mode
        
        self.visual_augmenter = VisualEffect(color_prob=0.2, contrast_prob=0.2,
            brightness_prob=0.2, sharpness_prob=0.2, autocontrast_prob=0.2,
            equalize_prob=0.2, solarize_prob=0.2)
        self.misc_augmenter = MiscEffect(multi_scale_prob=0.2, rotate_prob=0.2, flip_prob=0.2, crop_prob=0.2, translate_prob=0.2)    
            
    def on_epoch_end(self):
        np.random.shuffle(self.data)

    def __len__(self):
        """
        Number of batches for generator.
        """
        nums = len(self.data) // self.batch_size 
        if len(self.data) % self.batch_size == 0:
            return nums
        return nums + 1
    
    def get_group(self, batch):
        """
        Abstract function, need to implement
        """
        pass
    
    def __getitem__(self, idx):
        batch = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
        group_images, group_boxes, group_ids = self.get_group(batch)
        
        ##Augmentation
        if self.mode == 'train':
            group_images_aug, group_boxes_aug, group_ids_aug = [], [], []
            for image, boxes, class_id in zip(group_images, group_boxes, group_ids): 
                aug_image, aug_boxes = self.visual_augmenter(image.copy()), boxes.copy()
                aug_image, aug_boxes = self.misc_augmenter(aug_image, aug_boxes)
                if len(aug_boxes) != len(class_id):
#                     print("No equalization")
                      continue
                group_images_aug.append(aug_image)
                group_boxes_aug.append(aug_boxes)
                group_ids_aug.append(class_id)
            group_images = group_images_aug
            group_boxes = group_boxes_aug
            group_ids = group_ids_aug
        
        images, batch_hm, batch_wh, batch_reg = [], [], [], []
        for image, boxes, class_id in zip(group_images, group_boxes, group_ids):
            image, boxes = self.resizer(self.input_size, image, boxes) #512x512
            image = self.preprocess_image(image)
#             print(boxes)
            h, w = image.shape[:-1] 
            hm, wh, reg = self.compute_targets_each_image(boxes, class_id)
#             print(boxes, np.sum(hm), np.sum(wh), np.sum(reg), np.where(hm == 1.0))
            images.append(image)
            batch_hm.append(hm)
            batch_wh.append(wh)
            batch_reg.append(reg)
         
        outputs = np.concatenate([np.array(batch_wh, dtype=np.float32),
                                  np.array(batch_reg, dtype=np.float32),
                                  np.array(batch_hm, dtype=np.float32)
                                 ], -1)
        return np.array(images, dtype=np.float32), outputs
    
    def get_heatmap_per_box(self, heatmap, cls_id, ct_int, size):
        h, w = size
        radius = gaussian_radius((math.ceil(h), math.ceil(w)), min_overlap=0.7)
        radius = max(0, int(radius))
        heatmap[..., cls_id] = draw_gaussian_2(heatmap[...,cls_id], ct_int, radius)
        return heatmap

    def compute_targets_each_image(self, boxes, class_id):
        hm = np.zeros((self.output_size, self.output_size, self.num_classes), dtype=np.float32)
        whm = np.zeros((self.output_size, self.output_size, 2), dtype=np.float32)
        reg = np.zeros((self.output_size, self.output_size, 2), dtype=np.float32)
        for i, (box, cls_id) in enumerate(zip(boxes, class_id)):
            #scale box to output size
            xmin, ymin, xmax, ymax = [p/self.stride for p in box]
            h_box, w_box = ymax - ymin, xmax - xmin
            if h_box < 0 or w_box < 0:
                continue
            x_center, y_center = (xmax + xmin) / 2.0, (ymax + ymin) / 2.0
            x_center_floor, y_center_floor = int(np.floor(x_center)), int(np.floor(y_center))
            hm = self.get_heatmap_per_box(hm, cls_id, (x_center_floor, y_center_floor), (h_box, w_box))
            whm[x_center_floor, y_center_floor] = [w_box, h_box]
            reg[x_center_floor, y_center_floor] = x_center - x_center_floor, y_center - y_center_floor
#             print(ct - ct_int)
#         print(np.where(hm == 1))
#         print('------')
        return hm, whm, reg

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

#         image[..., 0] -= 103.939
#         image[..., 1] -= 116.779
#         image[..., 2] -= 123.68

        return image / 255.0
    
    def reverse_preprocess(self, image):
#         image[..., 0] += 103.939
#         image[..., 1] += 116.779
#         image[..., 2] += 123.68
        image *= 255.0
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_name2idx_mapper(self, path):
        '''
        from label name to index
        '''
        with open(path, 'r') as f:
            classes_dict = json.load(f)
        return classes_dict
                        
   