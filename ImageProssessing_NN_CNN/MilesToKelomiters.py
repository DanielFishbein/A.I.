#https://classroom.udacity.com/courses/ud187
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

miles = [1,   10,   23,   57,   40,   22,   2000]
km    = [1.6, 16.1, 37.0, 91.7, 64.4, 35.4, 3218.7]

l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([l0,l1,l2])
model.compile(loss = 'mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
history = model.fit(miles, km, epochs=1000, verbose=False)
print("Congrats on making it this far :)")

import matplotlib.pyplot as plt
plt.xlabel('Epoch number')
plt.ylabel('Loss magnitude')
plt.plot(history.history['loss'])
plt.show()
print(model.predict([5,0]))
