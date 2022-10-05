import sys
import os
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
import tensorflow as tf
from models.backbone.common_backbone import *
from models.backbone.CSPdarknet import CSPDarknet
from models.neck.neck import FPN, FPN_silu, PAN, yolov7_neck, FPN_ASF
from models.head import yolov6_head, centernet_head, Ihead
from models.common_layers import SPP

backbones = {'resnet50': ResNet50, 'resnet101': ResNet101, 'effb0': EffB0, 'densenet121': DenseNet121, 'cspdarknet': CSPDarknet}
necks = {'fpn':FPN, 'fpn_silu':FPN_silu, 'pan':PAN, 'v7neck':yolov7_neck, 'fpn_asf': FPN_ASF}
heads = {'v6head': yolov6_head, 'centernet': centernet_head, 'ihead':Ihead}

class Model:
    def __init__(self, backbone, neck, head, input_shape=(640, 640, 3), num_class=1, yolox_phi='tiny', weight_decay=5e-4):
        assert backbone in backbones.keys(), '%s backbone not supported'%backbone
        assert (neck == '') or (neck in necks.keys()), '%s neck not supported'%neck
        assert head in heads.keys(), '%s head not supported'%head
        
        self.input_shape = input_shape
        self.num_class = num_class
        #get backbone
        self.model_name = '%s_%s_%s'%(backbone, neck if neck != '' else 'None', head)
        if backbone == 'cspdarknet':
            self.flag = True
            self.backbone = CSPDarknet(input_shape, yolox_phi, weight_decay).build()
        else:
            self.flag = False
            self.backbone = backbones[backbone](input_shape).build()
        self.neck = necks[neck] if neck != '' else None
        self.head = heads[head]
    
    def build(self):
        inp = self.backbone.input
        if not self.flag:
            C2, C3, C4, C5 = self.backbone.output

            C5 = SPP(C5, 256)
            if self.neck is not None:
                x = self.neck([C2, C3, C4, C5], 256, return_fuse_only=True)
            else:
                x = C5
            out = self.head(x, 128, self.num_class)
        else:
            C3, C4, C5 = self.backbone.output
#             C3   = tf.keras.layers.UpSampling2D(size=(2, 2))(C3)
#             C4   = tf.keras.layers.UpSampling2D(size=(4, 4))(C4)
#             C5   = tf.keras.layers.UpSampling2D(size=(8, 8))(C5)
            kernel_reg = tf.keras.regularizers.l2(5e-4)
            C3 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(C3)
            C4 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=4, use_bias=False, padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(C4)
            C5 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=8, use_bias=False, padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(C5)
            fuse = tf.keras.layers.Concatenate()([C3, C4, C5])
            out  = self.head(fuse, 64, self.num_class)
           
        return tf.keras.models.Model(inputs=inp, outputs=out, name=self.model_name)
    
class MultiHeadModel:
    def __init__(self, backbone, neck, head, input_shape=(640, 640, 3), num_class=1, yolox_phi='tiny', weight_decay=5e-4):
        assert backbone in backbones.keys(), '%s backbone not supported'%backbone
        assert (neck == '') or (neck in necks.keys()), '%s neck not supported'%neck
        assert head in heads.keys(), '%s head not supported'%head
        
        self.input_shape = input_shape
        self.num_class = num_class
        #get backbone
        self.model_name = '%s_%s_%s'%(backbone, neck if neck != '' else 'None', head)
        if backbone == 'cspdarknet':
            self.flag = True
            self.backbone = CSPDarknet(input_shape, yolox_phi, weight_decay).build()
        else:
            self.flag = False
            self.backbone = backbones[backbone](input_shape).build()
        self.neck = necks[neck] if neck != '' else None
        self.head = heads[head]
    
    def build(self):
        inp = self.backbone.input
        if not self.flag:
            C2, C3, C4, C5 = self.backbone.output

            C5 = SPP(C5, 256)
            if self.neck is not None:
                _, [P3, P4, P5] = self.neck([C2, C3, C4, C5], 256, return_fuse_only=False)
                outs = [self.head(p, 64, self.num_class) for p in [P3, P4, P5]]
            else:
                outs = [self.head(p, 64, self.num_class) for p in [C3, C4, C5]]
        else:
            C3, C4, C5 = self.backbone.output
#             C3   = tf.keras.layers.UpSampling2D(size=(2, 2))(C3)
#             C4   = tf.keras.layers.UpSampling2D(size=(2, 2))(C4)
#             C5   = tf.keras.layers.UpSampling2D(size=(2, 2))(C5)
#             fuse = tf.keras.layers.Concatenate()([C3, C4, C5])
#             out  = self.head(fuse, 64, self.num_class)
            outs = [self.head(p, 64, self.num_class) for p in [C3, C4, C5]]
#             out1 = self.head(C3, 64, self.num_class)
#             out2 = self.head(C4, 64, self.num_class)
#             out3 = self.head(C5, 64, self.num_class)
#             print(out1.shape, out2.shape, out3.shape)
        return tf.keras.models.Model(inputs=inp, outputs=outs, name=self.model_name)
                 
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    input_shape = (640, 640, 3)
    model = Model('resnet50', 'fpn_asf', 'ihead', input_shape, num_class=3, yolox_phi='tiny', weight_decay=5e-4).build()
#     model = MultiHeadModel('cspdarknet', '', 'ihead', input_shape, num_class=3, yolox_phi='tiny', weight_decay=5e-4).build()
    model.summary()
    import numpy as np
    from time import time
    data = np.random.random((5,)+input_shape)
    for d in data:
        s = time()
        res = model(np.expand_dims(d, 0))
        print(res.shape, time()-s)
#         print(res[0].shape, res[1].shape, res[2].shape, time()-s)
    
    data = np.random.random((16,) + input_shape)
    s = time()
    res = model(data)
    print(time()-s)
         