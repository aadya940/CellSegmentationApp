import tensorflow as tf
import os
from keras import backend as K
import numpy as np

class UNet():
    def __init__(self, input_shape_):
        self.input_shape = input_shape_
    
    def dice_coef(self, y_true, y_pred, smooth=100):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice
    
    def dice_coef_loss(self, y_true, y_pred):
        return -self.dice_coef(y_true, y_pred)
    
    def iou(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        def f(y_true, y_pred):
            intersection = (y_true * y_pred).sum()
            union = y_true.sum() + y_pred.sum() - intersection
            x = (intersection + 1e-15) / (union + 1e-15)
            x = x.astype(np.float32)
            return x
        
        return tf.numpy_function(f, [y_true, y_pred], tf.float32)

    def buildModel(self):
        
        # Lets create the DownSampling Blocks 
        kernel_initializer =  'he_uniform'
         
        inputs = tf.keras.Input(shape = self.input_shape)
        
        # Block - 1
    
        s = inputs

        #Contraction path
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l1")(s)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l11")(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l2")(p1)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l22")(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l3")(p2)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l33")(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        p3 = tf.keras.layers.Dropout(0.2)(p3)

        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l4")(p3)
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l44")(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
        p4 = tf.keras.layers.Dropout(0.2)(p4)

        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l5")(p4)
        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l55")(c5)

        #Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name="l6")(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l66")(u6)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l666")(c6)

        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name="l7")(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l77")(u7)
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l777")(c7)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name="l8")(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l88")(u8)
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l888")(c8)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name="l9")(c8)
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l99")(u9)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation= "relu", kernel_initializer=kernel_initializer, padding='same', name="l999")(c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
        #compile model outside of this function to make it flexible. 
        model.summary()

        return model
    
    def CompileandSummarize(self, model_):
        model_.compile(optimizer = tf.keras.optimizers.Adam(), loss = self.dice_coef_loss, metrics = self.iou)


def load_model(path = os.path.join(os.getcwd(), 'package\src\my_model.h5')):
    '''
    Loads tf.model object from path

    Args:
        path: Path of the model in the file system
    '''
    mod_ = tf.keras.models.load_model(path, custom_objects= {'dice_coef_loss': UNet.dice_coef_loss, \
                                                             'iou': UNet.iou})
    return mod_