import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import tensorflow as tf
import keras

import numpy as np

#mco-mnist-lab
def make_model():
  return tf.keras.Sequential(
  [
      tf.keras.layers.Reshape(input_shape=(784,), target_shape=(28, 28, 1)),
      
      tf.keras.layers.Conv2D(input_shape=(28,28,1), kernel_size=3, filters=12, use_bias=False, padding='same'),
      tf.keras.layers.BatchNormalization(center=True, scale=False),
      tf.keras.layers.Activation('relu'),
      
      tf.keras.layers.Conv2D(kernel_size=6, filters=24, use_bias=False, padding='same', strides=2),
      tf.keras.layers.BatchNormalization(center=True, scale=False),
      tf.keras.layers.Activation('relu'),
      
      tf.keras.layers.Conv2D(kernel_size=6, filters=32, use_bias=False, padding='same', strides=2),
      tf.keras.layers.BatchNormalization(center=True, scale=False),
      tf.keras.layers.Activation('relu'),
      
      tf.keras.layers.Flatten(),
      
      tf.keras.layers.Dense(200, use_bias=False),
      tf.keras.layers.BatchNormalization(center=True, scale=False),
      tf.keras.layers.Activation('relu'),
      
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

class Recognise:
    
    def __init__(self):
      self.model = make_model()
      self.model = keras.models.load_model('mnist.h5')

    def convert(self,pil_img):      
      #pil image 280 x 280
      img = pil_img.resize((28, 28))
      self.img_28_28 = img
      #numpy array [0,255]
      pixels = np.array(img)
      #uint [0,255]
      pixels = pixels.astype('float32')
      #normalize [0,1]
      pixels /= 255.0
      #gray(alpha) from rgba
      pixels = pixels[:,:,3]
      #flatten: 28x28 = 784
      pixels = pixels.reshape(1,-1)

      return pixels

    def out(self,pil_img):
      x = self.convert(pil_img)
      #print(x.shape)
      #return ("not ready")
      #x: flattened (1,784), normalized [0,1] numpy array 
      out = self.model(x)
      out = out.numpy()
      #print("prediction",out.argmax(axis=1))
      prediction = out.argmax(axis=1)
      return prediction[0]

    def test(self):
      mnist = tf.keras.datasets.mnist

      (x_train, y_train), (x_test, y_test) = mnist.load_data()
      x_train, x_test = x_train / 255.0, x_test / 255.0   

      x = x_train.reshape(60000,-1)
      x = x[0].reshape(1,-1)
      y = y_train[0] 

      o = self.out(x)

      print(o)
      print(o.sum(axis=1))
      print(o.max(axis=1))
      print("target/label",y)
      print("prediction",o.argmax(axis=1))              




