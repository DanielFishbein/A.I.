#https://classroom.udacity.com/courses/ud187
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import sys
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#loading the dataset
#splitting the dataset into training data and testing data
#defining class names that will be used in training
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
class_names = ['0','1','2','3','4','5','6','7','8','9']

#finding the number of examples in the training set and the test set
num_train_examples = metadata.splits['train'].num_examples
num_test_examples  = metadata.splits['test'].num_examples

#function that normalizes the image values from 1->255 to 0->1
#returns the normalized image and corisponiding label
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images = images/255
    return images, labels


#maping labels to images for each dataset while also normalizing each image
train_dataset = train_dataset.map(normalize)
test_dataset  = test_dataset.map(normalize)

#fancy thing to make things run faster
train_dataset = train_dataset.cache()
test_dataset  = test_dataset.cache()

 #printing the first image in the test dataset
for image,label in test_dataset.take(1):
        break
image = image.numpy().reshape(28,28)
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()



#defining how the model is to be made (neurons and how they are connected)
model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape=(28,28,1)),
                            tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu),
                            tf.keras.layers.Dense(16, activation=tf.nn.sigmoid),
                            tf.keras.layers.Dense(10,activation=tf.nn.softmax)
                            ])

#building the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#number of images in each batch
#prepairing the training data by puting it in batches, shuffling the batches, and then repeating the batches
batch_size = 50
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

#training the model on the training data
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/batch_size))

#pulling data from the model after training
test_lost, test_accuracy = model.evaluate(test_dataset)
print('Accuracy on the dataset: {}'.format(test_accuracy))

#pulling the time
#and saving the NN so that i t can be pulled later
t = time.time()
export_path_keras = "NumbersNN_Demo_{}.h5".format(int(t))
print(export_path_keras)
model.save(export_path_keras)

#pulling the first 2 images and labels from the dest dataset
#calling the value of the image and the label
#guessing the number of each test image in the tests dataset
#takind the first prediction for each image
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
    print("Predictions Guess: {}".format(np.argmax(predictions[0])))
    print("Test Label: {}".format(test_labels[0]))

plt.show()
