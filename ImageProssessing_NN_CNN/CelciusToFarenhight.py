#https://classroom.udacity.com/courses/ud187
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


cels_q = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahr_a = np.array([-40,  14, 32, 46, 59, 72, 100],dtype=float)

#for i,c in enumerate(cels_q):
#    print(i,"{} degrees Celsius = {} degrees Farhenhight".format(c,fahr_a[i]))

L0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([L0])

#sometimes written as:
'''
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])
'''

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(cels_q, fahr_a,epochs=500, verbose=False)
print("Finished training the model")

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()
print(model.predict([100,0]))
print("These are the layer variables: {}".format(L0.get_weights()))
