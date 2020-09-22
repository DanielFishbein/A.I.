import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_datasets as tfds

#removes a harmless error message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#function that normaises the value of each pixel from 0->255 to 0->1
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images = images/255
    return images, labels


#loading the dataset
#identifying the test dataset
#for each item in the test dataset normalize it
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
test_dataset = dataset['test']

#normalize the images data
#shuffle the data set to within 1000 of it initial position
#get a batch of images of size 1
test_dataset  = test_dataset.map(normalize)
test_dataset = test_dataset.shuffle(1000)
test_dataset = test_dataset.batch(1)


#load a model
model = tf.keras.models.load_model("NumbersNN_Demo_1577727577.h5",
                                      custom_objects={'leaky_relu': tf.nn.leaky_relu})


#take the 1 image from the batch
for test_images, test_labels in test_dataset.take(1):

    #convert the image into a printable shape and format
    #print image
    image = test_images.numpy().reshape(28,28)
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.title("What the computer sees")
    plt.show()


    #predict the image
    #print the predictions highest value location
    predictions = model.predict(test_images)
    print("Prediction: {}".format(np.argmax(predictions)))
