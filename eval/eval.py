import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorflow as tf
import json
import xml.etree.ElementTree as ET

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)

    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68

    return image

def resize(size, image):
    h, w, c = image.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)
    padimg = np.zeros((size, size, c), image.dtype)
    padimg[:h, :w] = cv2.resize(image, (w, h))
    return padimg

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

def infer(model, image, score_threshold=0.01):
    input_size = 640
    h, w = image.shape[:2]
    image = resize(input_size, image)
#     image = cv2.resize(image, (512, 512))
    image = preprocess_image(image)

    inputs = tf.expand_dims(image, axis=0)
    # run network
    detections = model(inputs, training=False)[0].numpy()
    boxes, scores, class_ids = detections[..., :4], detections[..., 4], detections[..., 5]
    boxes = boxes * input_size
    boxes = rescale_boxes(input_size, (w, h), boxes)
    corrected_boxes, corrected_scores, corrected_classes = [], [], []
    for score, box, c in zip(scores, boxes, class_ids):
        if score < score_threshold:
            continue
        xmin, ymin, xmax, ymax = box[:4]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
#         xmin, xmax = int(xmin * w), int(xmax * w)
#         ymin, ymax = int(ymin * h), int(ymax * h)
        corrected_boxes.append([xmin, ymin, xmax, ymax])
        corrected_classes.append(c)
        corrected_scores.append(score)

    return corrected_boxes, scores, class_ids

def infer_TFAPI(model, image, score_threshold=0.001):
    
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = tf.expand_dims(image, axis=0)
    # run network
    output_dict = model(inputs)
    boxes = output_dict['detection_boxes'][0].numpy()
    scores = output_dict['detection_scores'][0].numpy()
    classes = output_dict['detection_classes'][0].numpy().astype('int32')
    corrected_boxes, corrected_scores, corrected_classes = [], [], []
    for score, box, c in zip(scores, boxes, classes):
        if score < score_threshold:
            continue
        ymin, xmin, ymax, xmax = box[:4]
        xmin, xmax = int(xmin * w), int(xmax * w)
        ymin, ymax = int(ymin * h), int(ymax * h)
        corrected_boxes.append([xmin, ymin, xmax, ymax])
        corrected_classes.append(c-1)
        corrected_scores.append(score)

    return corrected_boxes, corrected_scores, corrected_classes

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

def generate_coco_format_labels(img_path, label_path, name2idx_mapper, save_path):
    class_names = name2idx_mapper.keys()
    # for evaluation with pycocotools
    dataset = {"categories": [], "annotations": [], "images": []}
    for i, class_name in enumerate(class_names):
        dataset["categories"].append(
            {"id": i, "name": class_name, "supercategory": ""}
        )

    ann_id = 0
    print('Check annotations ...')
    items, filt_out = [], 0
    for name in os.listdir(img_path):
        lp = os.path.join(label_path, '.'.join(name.split('.')[:-1])+'.xml')
#         lp = os.path.join(label_path, name[:-3] + 'xml')
#         print(lp)
        if not os.path.exists(lp):
            print("No annotation for ", name)
            filt_out += 1
            continue
        items.append({'name':name, 'fp':os.path.join(img_path, name), 'lp':lp})
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
            cls_id = name2idx_mapper[obj_name]
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

def generate_coco_format_predict(img_path, model, save_path, score_threshold=0.01):
    """
    Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        generator: The generator for generating the evaluation data.
        model: The model to evaluate.
        threshold: The score threshold to use.
    """
    print('Generate COCO format for predictions')
    # start collecting results
    results = []
    for i, name in tqdm(enumerate(os.listdir(img_path))):
        if not ('jpg' in name or 'png' in name):
            continue
        fp = os.path.join(img_path, name)
        image = cv2.imread(fp)
        src_image = image.copy()
        boxes, scores, class_ids = infer(model, image, score_threshold)  
#         boxes, scores, class_ids = infer_TFAPI(model, image, score_threshold=0.001)
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
    for catId in coco_true.getCatIds():
        print('Evaluating for class index: ', catId)
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
#         coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.params.catIds = [catId]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_eval.stats

if __name__ == '__main__':
    import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    ## Eval Segmentations dataset
    
#     root = '/data2/sonnh/E2EObjectDetection/Segment_datasets/TestDataset'
#     datasets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
#     dataset = datasets[4]
#     print('EVALUATE FOR DATASET: ', dataset)
#     im_path = os.path.join(root, dataset, 'images')
#     lb_path = os.path.join(root, dataset, 'bboxes')
#     anno_json = os.path.join(root, dataset, 'annotations.json')
#     pred_json = os.path.join('/data2/sonnh/E2EObjectDetection/', '%s_preds.json'%dataset)
    
    ##Eval PolypsSet
    root = '/data2/sonnh/E2EObjectDetection/PolypsSet'
    dataset = 'test2019'
    im_path = os.path.join(root, dataset, 'images')
    lb_path = os.path.join(root, dataset, 'Annotation')
    anno_json = os.path.join(root, dataset, 'annotations.json')
    pred_json = os.path.join('/data2/sonnh/E2EObjectDetection/Centernet/trained_models/20220916', '%s_preds.json'%dataset)

#     NAME2IDX = {'polyp':0, 'adenomatous':0, 'hyperplastic':1}
    NAME2IDX = {'adenomatous':0, 'hyperplastic':1}
#     model = tf.saved_model.load('/data2/sonnh/E2EObjectDetection/Centernet/trained_models/polyp_tfapi_20220703/saved_model')
    model = tf.saved_model.load('/data2/sonnh/E2EObjectDetection/Centernet/trained_models/20220923/savedmodel')
    generate_coco_format_labels(im_path, lb_path, NAME2IDX, anno_json)
    generate_coco_format_predict(im_path, model, pred_json, score_threshold=0.001)
    evaluate_all(anno_json, pred_json)
    
    
    