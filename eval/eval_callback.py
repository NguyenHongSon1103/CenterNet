import sys
sys.path.append('/data/sonnh/E2EObjectDetection/Centernet')
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorflow as tf
import json
import xml.etree.ElementTree as ET
from models.decoder import decode
import os

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def parse_xml(xml):
    root = ET.parse(xml).getroot()
    objs = root.findall('object')
    boxes, ymins, obj_names = [], [], []
    for obj in objs:
        obj_name = obj.find('name').text
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        ymins.append(ymin)
        boxes.append([xmin, ymin, xmax, ymax])
        obj_names.append(obj_name)
    indices = np.argsort(ymins)
    boxes = [boxes[i] for i in indices]
    obj_names = [obj_names[i] for i in indices]
    return np.array(boxes, dtype=np.float), obj_names

def rescale_boxes(size, im_shape, boxes):
    w, h = im_shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    new_anns = []
    for box in boxes:
        xmin, ymin, xmax, ymax = [int(p/scale) for p in box]
        new_anns.append([xmin, ymin, xmax, ymax])
    return np.array(new_anns)

def generate_coco_format_predict_multi(generator, model, save_path, score_threshold=0.01):
    """
    Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        generator: The generator for generating the evaluation data.
        model: The model to evaluate.
        threshold: The score threshold to use.
    """
    # start collecting results
    results = []
    for i, (images, _) in tqdm(enumerate(generator)):
        output = model(images, training=False)[0]
        detections = decode(output, generator.num_classes, max_objects=100).numpy()
        batch_boxes, batch_scores, batch_class_ids = [], [], []
        batch = generator.get_batch(i)
        for b, detection in zip(batch, detections):
            raw_boxes, raw_scores, raw_class_ids = detection[..., :4], detection[..., 4], detection[..., 5]
            raw_boxes = raw_boxes * generator.input_size
            im_w, im_h = Image.open(b['im_path']).size
            raw_boxes = rescale_boxes(generator.input_size,  (im_w, im_h), raw_boxes)
            boxes, scores, class_ids = [], [], []
            for score, box, c in zip(raw_scores, raw_boxes, raw_class_ids):
                if score < score_threshold:
                    continue
                xmin, ymin, xmax, ymax = box[:4]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    #             xmin, xmax = int(xmin * w), int(xmax * w)
    #             ymin, ymax = int(ymin * h), int(ymax * h)
                boxes.append([xmin, ymin, xmax, ymax])
                class_ids.append(c)
                scores.append(score)
            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_class_ids.append(class_ids)
        
        names = generator.get_filenames(i)
        for boxes, scores, class_ids, name in zip(batch_boxes, batch_scores, batch_class_ids, names):
            # compute predicted labels and scores
            for box, score, class_id in zip(boxes, scores, class_ids):
                xmin, ymin, xmax, ymax = box
                w, h = max(0, xmax - xmin), max(0, ymax - ymin)  

                # append detection for each positively labeled class
                image_result = {
                    'image_id': os.path.splitext(name)[0],
                    'category_id': int(class_id),
                    'score': float(score),
                    'bbox': [xmin, ymin, w, h],
                }
                # append detection to results
                results.append(image_result)

    if not len(results):
        print('No testset found')
        return

    # write output
    json.dump(results, open(save_path, 'w'))
    print(f"Prediction to COCO format finished. Resutls saved in {save_path}")

def generate_coco_format_labels(generator, save_path):
    class_names = generator.name2idx.keys()
    # for evaluation with pycocotools
    dataset = {"categories": [], "annotations": [], "images": []}
    for i, class_name in enumerate(class_names):
        dataset["categories"].append(
            {"id": i, "name": class_name, "supercategory": ""}
        )

    ann_id = 0
    print('Check annotations ...')
    items, filt_out = [], 0
    for i, _ in enumerate(generator):
        batch = generator.get_batch(i)
        names = generator.get_filenames(i)
        for b, name in zip(batch, names):
            fp, lp = b['im_path'], b['lb_path']
            if not os.path.exists(lp):
                print("No annotation for ", name)
                filt_out += 1
                continue
            items.append({'name':name, 'fp':fp, 'lp':lp})
    print('Filter out %d images without labels'%filt_out)
    print('Total %d images with labels left'%len(items))
    for i, item in enumerate(items):
        img_id = os.path.splitext(item['name'])[0]
        img_w, img_h = Image.open(item['fp']).size
        dataset["images"].append(
            {
                "file_name": item['name'],
                "id": img_id,
                "width": img_w,
                "height": img_h,
            }
        )
        boxes, obj_names = parse_xml(item['lp'])
        for box, obj_name in zip(boxes, obj_names):
            x1, y1, x2, y2 = box
            # cls_id starts from 0
            cls_id = generator.name2idx[obj_name]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            dataset["annotations"].append(
                {
                    "area": h * w,
                    "bbox": [x1, y1, w, h],
                    "category_id": cls_id,
                    "id": ann_id,
                    "image_id": img_id,
                    "iscrowd": 0,
                    # mask
                    "segmentation": [],
                }
            )
            ann_id += 1

    with open(save_path, "w") as f:
        json.dump(dataset, f)
    print(f"Convert to COCO format finished. Resutls saved in {save_path}")

def generate_coco_format_predict(generator, model, save_path, score_threshold=0.01):
    """
    Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        generator: The generator for generating the evaluation data.
        model: The model to evaluate.
        threshold: The score threshold to use.
    """
    # start collecting results
    results = []
    for i, (images, _) in tqdm(enumerate(generator)):
        output = model(images, training=False)
        detections = decode(output, generator.num_classes, max_objects=100).numpy()
        batch_boxes, batch_scores, batch_class_ids = [], [], []
        batch = generator.get_batch(i)
        for b, detection in zip(batch, detections):
            raw_boxes, raw_scores, raw_class_ids = detection[..., :4], detection[..., 4], detection[..., 5]
            raw_boxes = raw_boxes * generator.input_size
            im_w, im_h = Image.open(b['im_path']).size
            raw_boxes = rescale_boxes(generator.input_size,  (im_w, im_h), raw_boxes)
            boxes, scores, class_ids = [], [], []
            for score, box, c in zip(raw_scores, raw_boxes, raw_class_ids):
                if score < score_threshold:
                    continue
                xmin, ymin, xmax, ymax = box[:4]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    #             xmin, xmax = int(xmin * w), int(xmax * w)
    #             ymin, ymax = int(ymin * h), int(ymax * h)
                boxes.append([xmin, ymin, xmax, ymax])
                class_ids.append(c)
                scores.append(score)
            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_class_ids.append(class_ids)
        
        names = generator.get_filenames(i)
        for boxes, scores, class_ids, name in zip(batch_boxes, batch_scores, batch_class_ids, names):
            # compute predicted labels and scores
            for box, score, class_id in zip(boxes, scores, class_ids):
                xmin, ymin, xmax, ymax = box
                w, h = max(0, xmax - xmin), max(0, ymax - ymin)  

                # append detection for each positively labeled class
                image_result = {
                    'image_id': os.path.splitext(name)[0],
                    'category_id': int(class_id),
                    'score': float(score),
                    'bbox': [xmin, ymin, w, h],
                }
                # append detection to results
                results.append(image_result)

    if not len(results):
        print('No testset found')
        return

    # write output
    json.dump(results, open(save_path, 'w'))
    print(f"Prediction to COCO format finished. Resutls saved in {save_path}")

def evaluate(anno_json, pred_json):
    # load results in COCO evaluation tool
    coco_true = COCO(anno_json)
    coco_pred = coco_true.loadRes(pred_json)

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
#     coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

def evaluate_all(anno_json, pred_json):
    # load results in COCO evaluation tool
    coco_true = COCO(anno_json)
    coco_pred = coco_true.loadRes(pred_json)

    # run COCO evaluation
    stats = []
    for catId in coco_true.getCatIds():
        with HiddenPrints():
            coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    #         coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval.params.catIds = [catId]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats.append(coco_eval.stats)
    return stats

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from generators.dataset_custom import get_polyp_test_generator
    from config.polyp_hparams import hparams
    import tensorflow_addons as tfa
    ## Eval Segmentations dataset
    ##Eval PolypsSet
    test_gen = get_polyp_test_generator(hparams)
    model_dir = 'trained_models/20221001'
    best_weights = 'weights_13_0.7588.h5'
    anno_json = os.path.join(model_dir, 'test_annotations.json')
    pred_json = os.path.join(model_dir, 'test_predictions.json')

    NAME2IDX = {'adenomatous':0, 'hyperplastic':1}
    with open(os.path.join(model_dir, 'model.json'), 'r') as f:
        model = tf.keras.models.model_from_json(f.read(),
                   custom_objects={'Addons>AdaptiveAveragePooling2D':tfa.layers.AdaptiveAveragePooling2D})
    weights_path = os.path.join(model_dir, best_weights)
    model.load_weights(weights_path)
    generate_coco_format_labels(test_gen, anno_json)
    generate_coco_format_predict(test_gen, model, pred_json, score_threshold=0.01)
    stats = evaluate_all(anno_json, pred_json)
    stats = np.array(stats)
    print('-'*5 + ' Test with model %s result '%weights_path + '-'*5)
    print('val_mAP@50    | ',end='')
    for i in range(test_gen.num_classes):
        print('Class %d: %8.3f | '%(i, stats[i][1]),end='')
    print('\n')
    print('val_mAP@50:95 | ',end='')
    for i in range(test_gen.num_classes):
        print('Class %d: %8.3f | '%(i, stats[i][0]),end='')
    print('\n')
    
    
    