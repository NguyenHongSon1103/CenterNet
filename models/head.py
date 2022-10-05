import tensorflow as tf
from models.common_layers import *
layers = tf.keras.layers

class ImplicitA:
    def __init__(self, channel, mean=0., std=.02):
        self.channel = channel
        self.mean = mean
        self.std = std
        initializer = tf.random_normal_initializer(mean=mean, stddev=std)
        self.implicit = tf.Variable(initializer(shape=(1, 1, 1, channel)), dtype=tf.float32, trainable=True)

    def __call__(self, x):
        return self.implicit + x

class ImplicitM:
    def __init__(self, channel, mean=1., std=.02):
        self.channel = channel
        self.mean = mean
        self.std = std
        initializer = tf.random_normal_initializer(mean=mean, stddev=std)
        self.implicit = tf.Variable(initializer(shape=(1, 1, 1, channel)), dtype=tf.float32, trainable=True)

    def __call__(self, x):
        return self.implicit * x

def yolov6_head(x, c=64, num_class=20):
    '''
                --> x1 --> out_hm
    x --> x --> 
                --> x2 --> out_wh
                       --> out_reg
    '''
    x = Conv(x, c*4, 1, 1)
    x = Conv(x, c*2, 1, 1)
    x1 = Conv(x, c, 3, 1)
    x2 = Conv(x, c, 3, 1)
    out_hm = layers.Conv2D(num_class, kernel_size=1, strides=1, padding='same',
                           bias_initializer=tf.keras.initializers.Constant(-4.6))(x1)
    out_hm = tf.nn.sigmoid(out_hm)
    
    out_wh = layers.Conv2D(2, kernel_size=1, strides=1, padding='same')(x2)
    
    out_reg = layers.Conv2D(2, kernel_size=1, strides=1, padding='same')(x2)
   
    return tf.concat([out_wh, out_reg, out_hm], -1)

def centernet_head(x, c=64, num_class=20):
    '''
                     --> out_hm
    x --> x --> x -- --> out_wh 
                     --> out_reg
    '''
    x = Conv(x, c*4, 3, 1)
    x = Conv(x, c*2, 3, 1)
    x = Conv(x, c, 3, 1)
    
    kernel_reg = tf.keras.regularizers.l2(5e-4)

    out_hm = Conv(x, c, 3, 1)
    out_hm = layers.Conv2D(num_class, 1, padding='same',
                           kernel_initializer='he_normal', kernel_regularizer=kernel_reg,
                           bias_initializer=tf.keras.initializers.Constant(-4.6))(out_hm)
    out_hm = tf.nn.sigmoid(out_hm)

    out_wh = Conv(x, c, 3, 1)
    out_wh = layers.Conv2D(2, 1, padding='same',
                           kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(out_wh)

    out_reg = Conv(x, c, 3, 1)
    out_reg = layers.Conv2D(2, 1, padding='same',
                           kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(out_reg)

    out = tf.concat([out_wh, out_reg, out_hm], -1)
    return out

def Ihead(x, c=64, num_class=20):
    ia, im = ImplicitA(c), ImplicitM(c)
    
    x = Conv(x, c*4, 3, 1)
    x = Conv(x, c*2, 3, 1)
    x = Conv(x, c, 3, 1)
    
    kernel_reg = tf.keras.regularizers.l2(5e-4)

    out_hm = Conv(x, c, 3, 1)
    out_hm = ia(out_hm)
    out_hm = Conv(out_hm, c, 3, 1)
    out_hm = im(out_hm)
    out_hm = layers.Conv2D(num_class, 1, kernel_initializer='he_normal', kernel_regularizer=kernel_reg,
                      bias_initializer=tf.keras.initializers.Constant(-4.6))(out_hm)
    out_hm = tf.nn.sigmoid(out_hm)

    out_wh = Conv(x, c, 3, 1)
    out_wh = ia(out_wh)
    out_wh = Conv(out_wh, c, 3, 1)
    out_wh = im(out_wh)
    out_wh = layers.Conv2D(2, 1, padding='same')(out_wh)

    out_reg = Conv(x, c, 3, 1)
    out_reg = ia(out_reg)
    out_reg = Conv(out_reg, c, 3, 1)
    out_reg = im(out_reg)
    out_reg = layers.Conv2D(2, 1, padding='same')(out_reg)

    out = tf.concat([out_wh, out_reg, out_hm], -1)
    return out
    
        