import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.python.keras.backend import dtype
import tensorflow as tf

class ImageProcess:
    __WIDTH  = 84
    __HEIGHT = 84
    # Constructor
    def __init__(self):
        self.shape  = (self.__WIDTH, self.__HEIGHT)
    def preprocess(self, image, debug = False):
        gray_image =tf.image.rgb_to_grayscale(image)
        resized = tf.image.resize(gray_image, (84,84), method='nearest').numpy()
        if debug == True:
            img_disp = np.concatenate([resized, resized, resized], axis = 2)
            plt.imshow(img_disp)
            plt.show()
        return resized
    def get_shape(self):
        return self.shape
