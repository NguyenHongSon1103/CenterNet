import sys
sys.path.append('/data2/sonnh/E2EObjectDetection/Centernet')
import tensorflow as tf
import numpy as np
from models.model import Model
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# backbones = {'resnet50': ResNet50, 'effb0': EffB0, 'cspdarknet53': CSPDarknet53, 'yolov7tiny':Yolov7tiny}
# necks = {'fpn':FPN, 'fpn_silu':FPN_silu, 'pan':PAN, 'v7neck':yolov7_neck, 'fpn_asf': FPN_ASF}
# heads = {'v6head': yolov6_head, 'centernet': centernet_head}
backbones = ['resnet50', 'effb0', 'cspdarknet53', 'yolov7tiny']
necks = ['fpn', 'fpn_silu', 'pan', 'v7neck', 'fpn_asf']
heads = ['v6head', 'centernet'] 
# total 4*5*2 = 40 options
inp_shape = (512, 512, 3)
data = np.random.random((16,)+inp_shape)

for backbone in backbones:
    if not backbone == 'effb0':
        continue
    for neck in necks:
#         if not neck == 'v7neck':
#             continue
        for head in heads:
            model = Model({'backbone':backbone, 'neck':neck, 'head':head}, inp_shape, 3).build()
            model_name = model.name
            model_params = np.sum([tf.size(w).numpy() for w in model.trainable_weights]) / 1e6
            model(data[0:1])
            s = time()
            res = model(data)
            print(model_name, '%.2f'%model_params, res.shape, time() - s)
            
print('test done')

'''
resnet50_fpn_v6head 22.02 (16, 128, 128, 7) 1.542619228363037
resnet50_fpn_centernet 21.56 (16, 128, 128, 7) 0.514700174331665
resnet50_fpn_silu_v6head 22.02 (16, 128, 128, 7) 0.2387852668762207
resnet50_fpn_silu_centernet 21.57 (16, 128, 128, 7) 0.19049835205078125
resnet50_pan_v6head 29.83 (16, 128, 128, 7) 0.6054830551147461
resnet50_pan_centernet 29.37 (16, 128, 128, 7) 0.2556774616241455
resnet50_v7neck_v6head 30.53 (16, 128, 128, 7) 0.6212751865386963
resnet50_v7neck_centernet 30.08 (16, 128, 128, 7) 0.49772167205810547
resnet50_fpn_asf_v6head 22.18 (16, 128, 128, 7) 0.3064906597137451
resnet50_fpn_asf_centernet 21.72 (16, 128, 128, 7) 0.3086833953857422
cspdarknet53_fpn_v6head 17.65 (16, 128, 128, 7) 0.7823259830474854
cspdarknet53_fpn_centernet 17.19 (16, 128, 128, 7) 0.4739103317260742
cspdarknet53_fpn_silu_v6head 17.65 (16, 128, 128, 7) 0.5476787090301514
cspdarknet53_fpn_silu_centernet 17.20 (16, 128, 128, 7) 0.48534154891967773
cspdarknet53_pan_v6head 25.45 (16, 128, 128, 7) 0.5811550617218018
cspdarknet53_pan_centernet 25.00 (16, 128, 128, 7) 0.5802464485168457
cspdarknet53_v7neck_v6head 26.16 (16, 128, 128, 7) 1.0691494941711426
cspdarknet53_v7neck_centernet 25.71 (16, 128, 128, 7) 0.8102223873138428
cspdarknet53_fpn_asf_v6head 17.81 (16, 128, 128, 7) 0.5997669696807861
cspdarknet53_fpn_asf_centernet 17.35 (16, 128, 128, 7) 0.6142585277557373
yolov7tiny_fpn_v6head 7.03 (16, 128, 128, 7) 0.2139296531677246
yolov7tiny_fpn_centernet 6.57 (16, 128, 128, 7) 0.16985368728637695
yolov7tiny_fpn_silu_v6head 7.03 (16, 128, 128, 7) 0.22572636604309082
yolov7tiny_fpn_silu_centernet 6.58 (16, 128, 128, 7) 0.2319192886352539
yolov7tiny_pan_v6head 13.30 (16, 128, 128, 7) 0.3756222724914551
yolov7tiny_pan_centernet 12.84 (16, 128, 128, 7) 0.2579786777496338
yolov7tiny_v7neck_v6head 15.61 (16, 128, 128, 7) 0.5099170207977295
yolov7tiny_v7neck_centernet 15.15 (16, 128, 128, 7) 0.49292445182800293
yolov7tiny_fpn_asf_v6head 7.19 (16, 128, 128, 7) 0.29648327827453613
yolov7tiny_fpn_asf_centernet 6.73 (16, 128, 128, 7) 0.29271793365478516

resnet50_fpn_v6head 22.02 (16, 128, 128, 7) 1.5476317405700684
resnet50_fpn_centernet 21.56 (16, 128, 128, 7) 0.5201027393341064
resnet50_fpn_silu_v6head 22.02 (16, 128, 128, 7) 0.2019662857055664
resnet50_fpn_silu_centernet 21.57 (16, 128, 128, 7) 0.21553611755371094
resnet50_pan_v6head 29.83 (16, 128, 128, 7) 0.5940461158752441
resnet50_pan_centernet 29.37 (16, 128, 128, 7) 0.24135732650756836
resnet50_v7neck_v6head 30.53 (16, 128, 128, 7) 0.6567590236663818
resnet50_v7neck_centernet 30.08 (16, 128, 128, 7) 0.517876386642456
resnet50_fpn_asf_v6head 22.18 (16, 128, 128, 7) 0.2760779857635498
resnet50_fpn_asf_centernet 21.72 (16, 128, 128, 7) 0.29556798934936523
cspdarknet53_fpn_v6head 17.65 (16, 128, 128, 7) 0.7833364009857178
cspdarknet53_fpn_centernet 17.19 (16, 128, 128, 7) 0.4646475315093994
cspdarknet53_fpn_silu_v6head 17.65 (16, 128, 128, 7) 0.5622012615203857
cspdarknet53_fpn_silu_centernet 17.20 (16, 128, 128, 7) 0.5103967189788818
cspdarknet53_pan_v6head 25.45 (16, 128, 128, 7) 0.6180791854858398
cspdarknet53_pan_centernet 25.00 (16, 128, 128, 7) 0.576796293258667
cspdarknet53_v7neck_v6head 26.16 (16, 128, 128, 7) 1.0825624465942383
cspdarknet53_v7neck_centernet 25.71 (16, 128, 128, 7) 0.8945086002349854
cspdarknet53_fpn_asf_v6head 17.81 (16, 128, 128, 7) 0.5128393173217773
cspdarknet53_fpn_asf_centernet 17.35 (16, 128, 128, 7) 0.6138947010040283
yolov7tiny_fpn_v6head 7.03 (16, 128, 128, 7) 0.22888994216918945
yolov7tiny_fpn_centernet 6.57 (16, 128, 128, 7) 0.16184043884277344
yolov7tiny_fpn_silu_v6head 7.03 (16, 128, 128, 7) 0.21739935874938965
yolov7tiny_fpn_silu_centernet 6.58 (16, 128, 128, 7) 0.22987914085388184
yolov7tiny_pan_v6head 13.30 (16, 128, 128, 7) 0.3917844295501709
yolov7tiny_pan_centernet 12.84 (16, 128, 128, 7) 0.255842924118042
yolov7tiny_v7neck_v6head 15.61 (16, 128, 128, 7) 0.5095088481903076
yolov7tiny_v7neck_centernet 15.15 (16, 128, 128, 7) 0.5207836627960205
yolov7tiny_fpn_asf_v6head 7.19 (16, 128, 128, 7) 0.29453325271606445
yolov7tiny_fpn_asf_centernet 6.73 (16, 128, 128, 7) 0.2983682155609131
'''