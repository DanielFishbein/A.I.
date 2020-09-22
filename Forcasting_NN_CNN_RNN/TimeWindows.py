#https://classroom.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/84c83c85-a543-44cd-837a-3c2614def701
#https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l08c04_time_windows.ipynb#scrollTo=y70nV0EI8OIZ
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

#creating a dataset
#window() = creating a moving window of 5 through the dataset
#flat_map() = arraying the windowed dataset
#map() = popping off the last eliment in the array but keaping it in a corisponding array
#shuffle() = shuffles the dataset
#batch() = how many of the arrays to pull at once
#prefetch() = fancy way of making things run faster

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print("x =", x.numpy())
    print("y =", y.numpy())

#Handy function that converts a dataset into a usable neural network dataset (does above but with a function)
def window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

series = [0,1,2,3,4,5,6,7,8,9]
window_size = 5
batch_size = 2

print(window_dataset(series, window_size,batch_size))
