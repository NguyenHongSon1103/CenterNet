import sys
sys.path.append('/data2/sonnh/E2EObjectDetection/Centernet')
import os
from generators.generator import Generator
# from generators.generator_multihead import Generator
from generators.utils import parse_xml, parse_coco_json
import cv2

class DatasetVOCFormat(Generator):
    '''
    Dataset provide as image - label (xml file)
    '''
    def __init__(self, data, hparams, mode='train'):
        super(DatasetVOCFormat, self).__init__(data, hparams, mode)
        self.names = hparams['names']
        #self.name2idx = {name:i for i, name in enumerate(self.names)}
        self.name2idx = hparams['name2idx']

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
    
    def get_filenames(self, idx):
        names = []
        batch = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
        for b in batch:
            name = b['im_path'].split('/')[-1]
            names.append(name)
        return names
    
    def get_batch(self, idx):
        return self.data[idx*self.batch_size:(idx+1)*self.batch_size]

    
def get_dummy_generator(hparams):
    root = '/data/sonnh8/E2EObjectDetection/YoloCenter/dummy_data'
    train_path = os.path.join(root, 'train')
    val_path = os.path.join(root, 'val')
    def get_data(path):
        data = []
        for name in os.listdir(path):
            if 'xml' in name:
                continue
            fp = os.path.join(path, name)
            xp = os.path.join(path, name[:-3] + 'xml')
            data.append({'im_path':fp, 'lb_path':xp})
        return data
    train_data, val_data = get_data(train_path), get_data(val_path)
    train_gen = DatasetVOCFormat(train_data, hparams, 'train')
    val_gen = DatasetVOCFormat(val_data, hparams, 'val')                          
    return train_gen, val_gen
    
# def get_polyp_generator(hparams):
#     root = '/data2/sonnh/E2EObjectDetection/PolypsSet'
#     train_path = os.path.join(root, 'train2019')
#     val_path = os.path.join(root, 'val2019')
    
#     train_data, val_data = [], []
    
#     for name in os.listdir(os.path.join(train_path, 'images')):
#         fp = os.path.join(train_path, 'images', name)
#         lp = os.path.join(train_path, 'Annotation', name[:-3] + 'xml')
#         if not os.path.exists(fp):
#             print('khong co anh')
#             continue
            
#         if not os.path.exists(lp):
#             print('khong co nhan')
#             continue
            
#         # Remove data with no label
#         boxes, names = parse_xml(lp)
#         if len(boxes) == 0:
#             continue
            
#         train_data.append({'im_path':fp, 'lb_path':lp})
    
#     for name in os.listdir(os.path.join(val_path, 'images')):
#         fp = os.path.join(val_path, 'images', name)
#         lp = os.path.join(val_path, 'Annotation', name[:-3] + 'xml')
#         if not os.path.exists(fp):
#             print('khong co anh')
#             continue
            
#         if not os.path.exists(lp):
#             print('khong co nhan')
#             continue
        
#         # Remove data with no label
#         boxes, names = parse_xml(lp)
#         if len(boxes) == 0:
#             continue
       
#         val_data.append({'im_path':fp, 'lb_path':lp})
            
#     train_gen = DatasetVOCFormat(train_data, hparams, 'train') 
#     val_gen = DatasetVOCFormat(val_data, hparams, 'val') 
    
#     return train_gen, val_gen

def get_polyp_generator(hparams):
    root = '/data/sonnh8/Polyps/PolypsSet_origin'
    train_path = os.path.join(root, 'train2019')
    val_path = os.path.join(root, 'val2019')
    
    train_data, val_data = [], []
    
    for name in os.listdir(os.path.join(train_path, 'Image')):
        fp = os.path.join(train_path, 'Image', name)
        lp = os.path.join(train_path, 'Annotation', name[:-3] + 'xml')
        if not os.path.exists(fp):
            print('khong co anh')
            continue
            
        if not os.path.exists(lp):
            print('khong co nhan')
            continue
            
        # Remove data with no label
        boxes, names = parse_xml(lp)
        if len(boxes) == 0:
            continue
            
        train_data.append({'im_path':fp, 'lb_path':lp})
    
    for sub in os.listdir(os.path.join(val_path, 'Image')):
        for name in os.listdir(os.path.join(val_path, 'Image', sub)):
            fp = os.path.join(val_path, 'Image', sub, name)
            lp = os.path.join(val_path, 'Annotation', sub, name[:-3] + 'xml')
            if not os.path.exists(fp):
                print('khong co anh')
                continue

            if not os.path.exists(lp):
                print('khong co nhan')
                continue

            # Remove data with no label
            boxes, names = parse_xml(lp)
            if len(boxes) == 0:
                continue

            val_data.append({'im_path':fp, 'lb_path':lp})
            
    train_gen = DatasetVOCFormat(train_data, hparams, 'train') 
    val_gen = DatasetVOCFormat(val_data, hparams, 'val') 
    
    return train_gen, val_gen

def get_polyp_test_generator(hparams):
    root = '/data/sonnh8/Polyps/PolypsSet_origin'
    test_path = os.path.join(root, 'test2019')
    test_data = []

    for sub in os.listdir(os.path.join(test_path, 'Image')):
        for name in os.listdir(os.path.join(test_path, 'Image', sub)):
            fp = os.path.join(test_path, 'Image', sub, name)
            lp = os.path.join(test_path, 'Annotation', sub, name[:-3] + 'xml')
            if not os.path.exists(fp):
                print('khong co anh')
                continue

            if not os.path.exists(lp):
                print('khong co nhan')
                continue

            # Remove data with no label
            boxes, names = parse_xml(lp)
            if len(boxes) == 0:
                continue

            test_data.append({'im_path':fp, 'lb_path':lp})
    
    hparams['batch_size'] = 8
    test_gen = DatasetVOCFormat(test_data, hparams, 'test') 
    
    return test_gen

if __name__ == '__main__':
    import os
#     from generators.dataset_custom import get_polyp_test_generator
    from config.polyp_multi_hparams import hparams
    test_gen = get_polyp_test_generator(hparams)
    for i, x in enumerate(test_gen):
        imgs, lbs = x
        print(lbs[0].shape, lbs[1].shape, lbs[2].shape)
        assert False

#     train_gen, val_gen = get_polyp_generator(hparams)
#     print(len(train_gen), len(val_gen))
#     for i, x in enumerate(train_gen):
#         if i % 100 == 0:
#             print(i)
    
