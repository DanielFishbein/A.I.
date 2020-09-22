#https://classroom.udacity.com/courses/ud187
#https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l07c01_saving_and_loading_models.ipynb#scrollTo=wC_AYRJU9NQe
from __future__ import absolute_import, division, print_function, unicode_literals
import time
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers
tfds.disable_progress_bar()

splits = tfds.Split.ALL.subsplit(weighted=(80,20))
splits, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True, split=splits)
(train_examples, validation_examples) = splits

def format_image(image,label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))//255
    return image, label

num_examples = info.splits['train'].num_examples
BATCH_SIZE = 32
IMAGE_RES = 224

train_batches = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                  input_shape=(IMAGE_RES,IMAGE_RES,3))
feature_extractor.trainable = False
model = tf.keras.Sequential([feature_extractor,
                             layers.Dense(2, activation='softmax')])
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 3
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

class_names = np.array(info.features['label'].names)
print(class_names)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]
predicted_class_names

print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  plt.title(predicted_class_names[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")

'''
Now that we've trained the model, we can save it as an HDF5 file, which is the format used by Keras.
Our HDF5 file will have the extension '.h5',
and it's name will correpond to the current time stamp.
'''

#save a model
t = time.time()

export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)

model.save(export_path_keras)


#load a model
relaoded = tf.keras.models.load_model(
                                     export_path_keras,
                                     #'custom_objects' tells keras how to load a 'hub.KerasLayer'
                                     custom_objects={'KerasLayer': hubKerasLayer})

reloaded.summary()

result_batch = model.predict(image_batch)
relpaded_result_batch = reloaded.predict(image_batch)

print((abs(result_batch - reloaded_result_batch)).max())

EPOCHS = 3
history = reloaded.fit(train_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches)
t = time.time()

export_path_sm="./{}".format(int(t))
print(export_path_sm)

tf.saved_model.save(model, export_path_sm)

reloaded_sm = tf.saved_model.load(export_path_sm)
reload_sm_result_batch = reloaded_sm(image_batch, training=False).numpy()
print((abs(result_batch - reload_sm_result_batch)).max())

t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)
tf.saved_model.save(model, export_path_sm)

reload_sm_keras = tf.keras.models.load_model(export_path_sm,
                                             custom_objects={'KerasLayer': hub.KerasLayer})
reload_sm_keras.summary()

result_batch = model.predict(image_batch)
reload_sm_keras_result_batch = reload_sm_keras.predict(image_batch)
