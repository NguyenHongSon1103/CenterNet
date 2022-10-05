import tensorflow as tf
import tensorflow.keras.layers as L
import sys
sys.path.append('/data2/sonnh/E2EObjectDetection/Centernet')
from losses import loss

class ModelBuilder:
    def __init__(self, hparams):
        self.num_classes = hparams['num_classes']
        self.backbone_type = 'resnet50'
        self.input_size = hparams['input_size']
        self.max_objects = hparams['max_objects']
        self.use_init_bias = True
        
        assert self.backbone_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        self.output_size = self.input_size // 4
    
        self.image_input = L.Input(shape=(self.input_size, self.input_size, 3))
        
        if self.backbone_type == 'resnet50':
            self.backbone =  tf.keras.applications.ResNet50(include_top=False, input_tensor=self.image_input)
#             self.out_layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
            self.out_layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block3_out', 'conv5_block1_out']
        else:
            self.backbone =  tf.keras.applications.ResNet101(include_top=False, input_tensor=self.image_input)
            self.out_layer_names = ['']
        # Resnet18, 34 and 152 currently not support 
    
    def fpn(self, features):
        C2, C3, C4, C5 = features
        in2 = L.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2) # 160x160
        in3 = L.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3) # 80x40
        in4 = L.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4) # 40x40
        in5 = L.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5) # 20x20
        # 1 / 32 * 8 = 1 / 4
        P5 = L.UpSampling2D(size=(8, 8))(
            L.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5)) # 160x160
        # 1 / 16 * 4 = 1 / 4
        out4 = L.Add()([in4, L.UpSampling2D(size=(2, 2))(in5)]) # 40x40
        P4 = L.UpSampling2D(size=(4, 4))(
            L.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4)) # 160x160
        # 1 / 8 * 2 = 1 / 4
        out3 = L.Add()([in3, L.UpSampling2D(size=(2, 2))(out4)]) # 80x80
        P3 = L.UpSampling2D(size=(2, 2))(
            L.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3)) # 160x160
        # 1 / 4
        P2 = L.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(
            L.Add()([in2, L.UpSampling2D(size=(2, 2))(out3)])) # 160x160
        # (b, /4, /4, 256)
        fuse = L.Concatenate()([P2, P3, P4, P5])
        return fuse
    
    def head(self, x):
        kernel_reg = tf.keras.regularizers.l2(5e-4)

        # hm header
#         y1 = L.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x)
        y1 = L.Conv2D(64, 3, padding='same')(x)
        y1 = L.BatchNormalization()(y1)
        y1 = L.ReLU()(y1)
        if self.use_init_bias:
            y1 = L.Conv2D(self.num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=kernel_reg, bias_initializer=tf.keras.initializers.Constant(-4.6))(y1)
        else:
            y1 = L.Conv2D(self.num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(y1)
        y1 = tf.nn.sigmoid(y1)

        # wh header
#         y2 = L.Conv2D(64, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x)
        y2 = L.Conv2D(64, 3, padding='same')(x)
        y2 = L.BatchNormalization()(y2)
        y2 = L.ReLU()(y2)
#         y2 = L.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(y2)
        y2 = L.Conv2D(2, 1, padding='same')(y2)

        # reg header
#         y3 = L.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x)
        y3 = L.Conv2D(64, 3, padding='same')(x)    
        y3 = L.BatchNormalization()(y3)
        y3 = L.ReLU()(y3)
#         y3 = L.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(y3)
        y3 = L.Conv2D(2, 1, padding='same')(y3)
        #concat: wh, reg, hm --> easy to extract heatmap if change num of classes
        out = tf.concat([y2, y3, y1], -1)
        return out
        
    def resnet(self):
        C2, C3, C4, C5 = [self.backbone.get_layer(l).output for l in self.out_layer_names]
        #b, 16, 16, 2048
        x = C5 
        # decoder
        num_filters = 256
        for i in range(3):
            num_filters = num_filters // pow(2, i)
            ## Note: Có thể sử dụng Deformable conv ở đây ##
            x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3,
                                    strides=1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            
            ## Sử dụng initializer mặc định
            x = L.Conv2DTranspose(num_filters, 3, strides=2, use_bias=False, padding='same')(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)

        out = self.head(x)    
        model = tf.keras.Model(inputs=self.image_input, outputs=out)
        return model
    
    def resnet_fpn(self):
        C2, C3, C4, C5 = [self.backbone.get_layer(l).output for l in self.out_layer_names]
        #b, 16, 16, 2048
        x = self.fpn([C2, C3, C4, C5])
        # decoder
        num_filters = 256
        kernel_reg = tf.keras.regularizers.l2(5e-4)
        for i in range(3):
            num_filters = num_filters // pow(2, i)
            ## Note: Có thể sử dụng Deformable conv ở đây ##
            x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3,
                                    strides=1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            
            ## Sử dụng initializer mặc định
            ## Thay Conv2D transpose bằng conv2d bình thường do fpn đã đưa ra feature map stride 1/4
            x = L.Conv2D(num_filters, 3, strides=1, use_bias=False, padding='same')(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)

        out = self.head(x)    
        model = tf.keras.Model(inputs=self.image_input, outputs=out)
        return model
    
if __name__ == '__main__':
    import numpy as np
    from time import time
    import os
    from config.polyp_hparams import hparams
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = ModelBuilder(hparams).resnet_fpn()
    model.summary()
    
    # Test inference time and output shape
    images = np.random.random((5, 512, 512, 3))
    res = model(images)
    print(res.shape)
    


    
