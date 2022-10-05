import tensorflow as tf
'''
 Some backbones pretrained with imagenet dataset
'''

class Backbone:
    def __init__(self, input_shape):
        self.image_input = tf.keras.layers.Input(shape=input_shape)
        self.backbone = None
        self.out_layer_names = []
    
    def build(self):
        C2, C3, C4, C5 = [self.backbone.get_layer(l).output for l in self.out_layer_names]
        return tf.keras.models.Model(inputs=self.image_input, outputs=[C2, C3, C4, C5])
        
class ResNet50(Backbone):
    def __init__(self, input_shape):
        super(ResNet50, self).__init__(input_shape)
        self.backbone =  tf.keras.applications.ResNet50(include_top=False, input_tensor=self.image_input)
        self.out_layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block3_out', 'conv5_block1_out']
        
class ResNet101(Backbone):
    def __init__(self, input_shape):
        super(ResNet101, self).__init__(input_shape)
        self.backbone =  tf.keras.applications.ResNet101(include_top=False, input_tensor=self.image_input)
        self.out_layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block3_out', 'conv5_block1_out']
        
class DenseNet121(Backbone):
    def __init__(self, input_shape):
        super(DenseNet121, self).__init__(input_shape)
        self.backbone =  tf.keras.applications.DenseNet121(include_top=False, input_tensor=self.image_input)
        self.out_layer_names = ['conv2_block6_concat', 'conv3_block12_concat', 'conv4_block24_concat', 'conv5_block9_concat']
    
class EffB0(Backbone):
    def __init__(self, input_shape):
        super(EffB0, self).__init__(input_shape)
        self.backbone =  tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=self.image_input)
        self.out_layer_names = ['block2a_activation', 'block3a_activation', 'block5a_activation', 'block6a_activation']

if __name__ == '__main__':
    model = ResNet50((640, 640, 3)).build()
    model.summary()
    outs = model.output
    for out in outs:
        print(out.shape)