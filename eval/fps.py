import tensorflow as tf
import numpy as np
import argparse
from time import time
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--savedmodel', type=str, required=True, help='Path to the savedmodel')
parser.add_argument('--size', type=int, default=512, help='Path to the h5')

args = parser.parse_args()

model = tf.saved_model.load(args.savedmodel)

data = np.random.random((8, args.size, args.size, 3)).astype('float32')

model(data[:1])

print('*'*5 + 'STARTING EVALUATE INFERENCE TIME' + '*'*5)
print('Test case 1: batch size = 1 ==> ',end='')
avg = []
for d in data:
    s = time()
    model(tf.expand_dims(d, 0))
    avg.append(time()-s)
mean = np.mean(avg)
print('Mean_time: %.3f ==> FPS: %.1f'%(mean, 1/mean))

print('Test case 2: batch size = 8 ==> ',end='')
avg = []
for i in range(5):
    s = time()
    model(data)
    avg.append(time()-s)
mean = np.mean(avg)
print('Mean_time: %.3f ==> FPS: %.1f'%(mean, 8/mean))