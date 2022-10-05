import sys
sys.path.append('/data2/sonnh/E2EObjectDetection/Centernet')
import os
from generators.generator import Generator
from generators.utils import parse_xml, parse_coco_json
import cv2
 
class COCOGenerator(Generator):
    def __init__(self, data, hparams, mode='train'):
        hparams['num_classes'] = 80
        super(COCOGenerator, self).__init__(data, hparams, mode)
        self.c80_to_c90 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                           58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
                           88, 89, 90]
    
    def get_group(self, batch):
        group_images, group_boxes, group_ids = [], [], []
        for b in batch:
            image = cv2.imread(b['im_path'])
            boxes, class_ids = b['boxes'], b['class_id']
            class_ids = [self.c80_to_c90.index(c) for c in class_ids]
            
            ## Convert box from coco format to voc format:
            boxes = [[xmin, ymin, xmin+w, ymin+h] for (xmin, ymin, w, h) in boxes]

            if len(boxes) == 0:
                continue
            
            if image is None:
                continue
            group_images.append(image)
            group_boxes.append(boxes)
            group_ids.append(class_ids)
        
        return group_images, group_boxes, group_ids
        
def get_coco_generator(hparams):
    root = '/data2/sonnh/COCO2017'
    train_data, _ = parse_coco_json(os.path.join(root, 'annotations', 'instances_train2017.json'))
    for d in train_data:
        d['im_path'] = os.path.join(root, 'train', 'images', d['filename'])

    val_data, _ = parse_coco_json(os.path.join(root, 'annotations', 'instances_val2017.json'))
    for d in val_data:
        d['im_path'] = os.path.join(root, 'val', 'images', d['filename'])
        
    train_gen = COCOGenerator(train_data, hparams, 'train')
    val_gen = COCOGenerator(val_data, hparams, 'val')
    return train_gen, val_gen

class VOCGenerator(Generator):
    def __init__(self, data, hparams, mode='train'):
        hparams['num_classes'] = 20
        super(VOCGenerator, self).__init__(data, hparams, mode)
        self.name2idx = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3,
                "bottle": 4, "bus": 5, "car": 6, "cat": 7, "chair": 8,
                "cow": 9, "diningtable": 10, "dog": 11, "horse": 12,
                "motorbike": 13, "person": 14, "pottedplant": 15,
                "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19}
        
    def get_group(self, batch):
        group_images, group_boxes, group_ids = [], [], []
        for b in batch:
            image = cv2.imread(b['im_path'])
            boxes, names = parse_xml(b['lb_path'])
            class_id = [self.name2idx[n] for n in names]

            if len(boxes) == 0:
                continue
            
            if image is None:
                continue
            group_images.append(image)
            group_boxes.append(boxes)
            group_ids.append(class_id)
        
        return group_images, group_boxes, group_ids
    
def get_voc_generator(hparams):
    root07 = '/data2/sonnh/E2EObjectDetection/VOC2007'
    root12 = '/data2/sonnh/E2EObjectDetection/VOC2012'
    
    def get_listnames(root, part='train'):
        with open(os.path.join(root, 'ImageSets', 'Main', part+'.txt'), 'r') as f:
            names = f.read().split('\n')
        names = [name + '.jpg' for name in names if name != '']
        return names
    
    def get_data(root, names):
        im_path = os.path.join(root, 'JPEGImages')
        lb_path = os.path.join(root, 'Annotations')
        data = []
        for name in names:
            ip = os.path.join(im_path, name)
            lp = os.path.join(lb_path, name[:-3] + 'xml')
            if not os.path.exists(ip):
#                 print('không thấy ảnh')
                continue
            if not os.path.exists(lp):
#                 print('Không thấy nhãn')
                continue
            data.append({'im_path':ip, 'lb_path':lp})
        return data
    
    train_names_12 = get_listnames(root12, 'train')
    train_names_07 = get_listnames(root07, 'train')
    val_names_12 = get_listnames(root12, 'val')
    val_names_07 = get_listnames(root07, 'val')

    train_data = get_data(root12, train_names_12) + get_data(root12, val_names_12) + get_data(root07, train_names_07)
#     train_data = get_data(root07, train_names_07)
    val_data = get_data(root07, val_names_07)
            
    train_gen = VOCGenerator(train_data, hparams, 'train') 
    val_gen = VOCGenerator(val_data, hparams, 'val') 
    
    return train_gen, val_gen

if __name__ == '__main__':
    import yaml
    with open('config/model.yaml', 'r') as f:
        hparams = yaml.safe_load(f)
    hparams['batch_size'] = 8
    train_gen, val_gen = get_voc_generator(hparams)
    print(len(train_gen), len(val_gen))
    for i, x in enumerate(train_gen):
        print(i)
        assert False
 