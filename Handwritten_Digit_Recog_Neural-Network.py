import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform import _pywrap_tf2
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)  = mnist.load_data()

x_train = tf.keras.util.normalize(x_train,axis=1)
x_test = tf.keras.util.normalize(x_test,axis=1)

model   = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(unit=128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(unit=128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(unit=10,activation = tf.nn.softmax))

model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.fit(x_train,y_train,epocs=3)


loss,accuracy = model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

model.save('Digits.model')

