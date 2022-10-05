import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from time import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from models.backbone.CSPdarknet import Focus, SiLU
from models.decoder import decode

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-j', '--json', type=str, required=True, help='Path to the json')
parser.add_argument('-h5', '--h5', type=str, required=True, help='Path to the h5')
parser.add_argument('-s', '--savedmodel', type=str, required=True, help='Path to the savedmodel')
args = parser.parse_args()


with open(args.json, 'r') as f:
    model = tf.keras.models.model_from_json(f.read(), custom_objects={'Focus':Focus, 'SiLU':SiLU, 'Addons>AdaptiveAveragePooling2D':tfa.layers.AdaptiveAveragePooling2D})
model.load_weights(args.h5)

out = model.output

decode_inp = tf.keras.layers.Input(shape=(None, None, None)) 
decode_out = decode(decode_inp, None, 10)
decode_model = tf.keras.models.Model(decode_inp, decode_out)

detections = decode_model(out)

model = tf.keras.Model(inputs=model.input, outputs=detections)
model.summary()
model.save(args.savedmodel)

print('Export done to %s !'%args.savedmodel)