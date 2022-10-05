import tensorflow as tf
from models.common_layers import * 
layers = tf.keras.layers
from models.neck.asf import ScaleFeatureSelection
    
def FPN(features, c=256, return_fuse_only=True):
    C2, C3, C4, C5 = features
    inner_c = c // 4
    in2 = layers.Conv2D(c, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2) # 160x160
    in3 = layers.Conv2D(c, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3) # 80x40
    in4 = layers.Conv2D(c, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4) # 40x40
    in5 = layers.Conv2D(c, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5) # 20x20
    # 1 / 32 * 8 = 1 / 4
    P5 = layers.UpSampling2D(size=(8, 8))(
        layers.Conv2D(inner_c, (3, 3), padding='same', kernel_initializer='he_normal')(in5)) # 160x160
    # 1 / 16 * 4 = 1 / 4
    out4 = layers.Add()([in4, layers.UpSampling2D(size=(2, 2))(in5)]) # 40x40
    P4 = layers.UpSampling2D(size=(4, 4))(
        layers.Conv2D(inner_c, (3, 3), padding='same', kernel_initializer='he_normal')(out4)) # 160x160
    # 1 / 8 * 2 = 1 / 4
    out3 = layers.Add()([in3, layers.UpSampling2D(size=(2, 2))(out4)]) # 80x80
    P3 = layers.UpSampling2D(size=(2, 2))(
        layers.Conv2D(inner_c, (3, 3), padding='same', kernel_initializer='he_normal')(out3)) # 160x160
    # 1 / 4
    P2 = layers.Conv2D(inner_c, (3, 3), padding='same', kernel_initializer='he_normal')(
        layers.Add()([in2, layers.UpSampling2D(size=(2, 2))(out3)])) # 160x160
    # (b, /4, /4, 256)
    fuse = layers.Concatenate()([P2, P3, P4, P5])
    if return_fuse_only:
        return fuse
    return fuse, [P3, P4, P5]

def FPN_silu(features, c=256, return_fuse_only=True):
    C2, C3, C4, C5 = features
    inner_c = c // 4
    in2 = Conv(C2, c, 1, 1)
    in3 = Conv(C3, c, 1, 1)
    in4 = Conv(C4, c, 1, 1)
    in5 = Conv(C5, c, 1, 1)
    # 1 / 32 * 8 = 1 / 4
    P5 = layers.UpSampling2D(size=(8, 8))(Conv(in5, inner_c, 3, 1)) # 160x160
    # 1 / 16 * 4 = 1 / 4
    out4 = layers.Add()([in4, layers.UpSampling2D(size=(2, 2))(in5)]) # 40x40
    P4 = layers.UpSampling2D(size=(4, 4))(Conv(out4, inner_c, 3, 1)) # 160x160
    # 1 / 8 * 2 = 1 / 4
    out3 = layers.Add()([in3, layers.UpSampling2D(size=(2, 2))(out4)]) # 80x80
    P3 = layers.UpSampling2D(size=(2, 2))(Conv(out3, inner_c, 3, 1)) # 160x160
    # 1 / 4
    out3 = layers.Add()([in2, layers.UpSampling2D(size=(2, 2))(out3)]) 
    P2 = Conv(out3, inner_c, 3, 1) # 160x160
    # (b, /4, /4, 256)
    fuse = layers.Concatenate()([P2, P3, P4, P5])
    if return_fuse_only:
        return fuse
    return fuse, [P3, P4, P5]

def FPN_ASF(features, c=256, return_fuse_only=False):
    C2, C3, C4, C5 = features
    inner_c = c // 4
    in2 = Conv(C2, c, 1, 1)
    in3 = Conv(C3, c, 1, 1)
    in4 = Conv(C4, c, 1, 1)
    in5 = Conv(C5, c, 1, 1)
    # 1 / 32 * 8 = 1 / 4
    P5 = layers.UpSampling2D(size=(8, 8))(Conv(in5, inner_c, 3, 1)) # 160x160
    # 1 / 16 * 4 = 1 / 4
    out4 = layers.Add()([in4, layers.UpSampling2D(size=(2, 2))(in5)]) # 40x40
    P4 = layers.UpSampling2D(size=(4, 4))(Conv(out4, inner_c, 3, 1)) # 160x160
    # 1 / 8 * 2 = 1 / 4
    out3 = layers.Add()([in3, layers.UpSampling2D(size=(2, 2))(out4)]) # 80x80
    P3 = layers.UpSampling2D(size=(2, 2))(Conv(out3, inner_c, 3, 1)) # 160x160
    # 1 / 4
    out3 = layers.Add()([in2, layers.UpSampling2D(size=(2, 2))(out3)]) 
    P2 = Conv(out3, inner_c, 3, 1) # 160x160
    # (b, /4, /4, 256)
    fuse = layers.Concatenate()([P2, P3, P4, P5])
    
    asf = ScaleFeatureSelection(c, inner_c, 4, 'scale_channel_spatial')
    fuse = asf(fuse, [P2, P3, P4, P5])
    if return_fuse_only:
        return fuse
    return fuse, [P3, P4, P5]

def PAN(features, c=256, return_fuse_only=True):
    _, C3, C4, C5 = features
    C5 = Conv(C5, c, 1, 1)
    C5_up = layers.UpSampling2D(size=(2, 2))(C5)
    C4 = Concat([C4, C5_up], -1)
    C4 = Conv(C4, c, 3, 1)
    C4 = Conv(C4, c, 1, 1)
    C4_up = layers.UpSampling2D(size=(2, 2))(C4)
    C3= Concat([C3, C4_up], -1)
    
    P3 = Conv(C3, c, 3, 1)
    x = Conv(P3, c, 3, 2)
    x = Concat([x, C4], -1)
    P4 = Conv(x, c, 3, 1)
    x = Conv(P4, c, 3, 2)
    x = Concat([x, C5], -1)
    P5 = Conv(x, c, 3, 1)
    
    P3   = tf.keras.layers.UpSampling2D(size=(2, 2))(P3)
    P4   = tf.keras.layers.UpSampling2D(size=(4, 4))(P4)
    P5   = tf.keras.layers.UpSampling2D(size=(8, 8))(P5)
    fuse = tf.keras.layers.Concatenate()([P3, P4, P5])
    
    if return_fuse_only:
        return fuse
        
    return fuse, [P3, P4, P5]

def DenseBlock(x, c):
    inter_c = c // 2
    x1 = Conv(x, c, 1, 1)
    x2 = Conv(x, c, 1, 1)
    x11 = Conv(x1, inter_c, 3, 1)
    x12 = Conv(x11, inter_c, 3, 1)
    x13 = Conv(x12, inter_c, 3, 1)
    x14 = Conv(x13, inter_c, 3, 1)
    x = Concat([x1, x2, x11, x12, x13, x14], -1)
    x = Conv(x, c, 1, 1)
    return x

def PANConvBlock(x, x2, c):
    x = Conv(x, c, 1, 1)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x2 = Conv(x2, c, 1, 1)
    x = Concat([x2, x], -1)
    
    x = DenseBlock(x, c)
    return x

def PANPoolingBlock(x, x2, c):
    y = MP(x)
    y = Conv(y, c, 1, 1)
    z = Conv(x, c, 1, 1)
    z = Conv(z, c, 3, 2)
    x = Concat([z, y, x2], -1)
    
    x = DenseBlock(x, 2*c)
    return x
    
def yolov7_neck(features, c=256, return_fuse_only=True):
    _, P3, P4, P5 = features
    inner_c = c // 2
    x63 = PANConvBlock(P5, P4, c)
    x75 = PANConvBlock(x63, P3, inner_c)
    x88 = PANPoolingBlock(x75, x63, inner_c)
    x101 = PANPoolingBlock(x88, P5, c)
    
    x75   = tf.keras.layers.UpSampling2D(size=(2, 2))(x75)
    x88   = tf.keras.layers.UpSampling2D(size=(4, 4))(x88)
    x101   = tf.keras.layers.UpSampling2D(size=(8, 8))(x101)
    fuse = tf.keras.layers.Concatenate()([x75, x88, x101])
    if return_fuse_only:
        return fuse
    return fuse, [x75, x88, x101]
    
if __name__ == '__main__':
    import numpy as np
    C3 = np.random.random((2, 80, 80, 3))
    C4 = np.random.random((2, 40, 40, 3))
    C5 = np.random.random((2, 20, 20, 3))
    res = PAN([C3, C4, C5], 64)
    print(res.shape)