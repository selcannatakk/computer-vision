import pathlib
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
# import keras

from VGG16_model import model


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

df = pd.read_csv("../data/fire_smoke_dataset/fire_smoke_labels_1.csv")
# print(df.head())

LABELS = ["fire", "smoke"]

# Prepare Data Pipeline by using tf.data

data_dir = pathlib.Path("../data/fire_smoke_dataset")
filenames = list(data_dir.glob('resize_images/*.jpg'))
fnames=[]
for fname in filenames:
  fname = str(fname)
  fname = fname.replace("\\", "/")
  fnames.append(fname)


ds_size = len(fnames)
print("Number of images in folders: ", ds_size)

number_of_selected_samples = 2000
filelist_ds = tf.data.Dataset.from_tensor_slices(fnames[:number_of_selected_samples])

ds_size = filelist_ds.cardinality().numpy()
print("Number of selected samples for dataset: ", ds_size)


def get_label(file_path):
  parts = tf.strings.split(file_path, '/')
  file_name= parts[-1]
  labels= df[df["Filenames"] == file_name][LABELS].to_numpy().squeeze()
  return tf.convert_to_tensor(labels)


# Let's resize and scale the images so that we can save time in training
IMG_WIDTH, IMG_HEIGHT = 64, 64


def decode_img(img):
  #color images
  img = tf.image.decode_jpeg(img, channels=3)
  #convert unit8 tensor to floats in the [0,1]range
  img = tf.image.convert_image_dtype(img, tf.float32)
  #resize
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


# Combine the images with labels
def combine_images_labels(file_path: tf.Tensor):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


# Decide the train-test split
train_ratio = 0.80
ds_train = filelist_ds.take(ds_size * train_ratio)
ds_test = filelist_ds.skip(ds_size * train_ratio)

# Decide the batch size
BATCH_SIZE = 64

# Pre-process all the images
ds_train = ds_train.map(lambda x: tf.py_function(func=combine_images_labels,
            inp=[x], Tout=(tf.float32, tf.int64)),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

ds_test = ds_test.map(lambda x: tf.py_function(func=combine_images_labels,
          inp=[x], Tout=(tf.float32,tf.int64)),
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False)

# Convert multi-hot labels to string labels
# onehot değerlerinden label(string) e ulaşıyoruz
def covert_onehot_string_labels(label_string, label_onehot):
  labels=[]
  for i, label in enumerate(label_string):
     if label_onehot[i]:
       labels.append(label)
  if len(labels) == 0:
    labels.append("NONE")
  return labels


# Show some samples from the data pipeline
def show_samples(dataset):
  fig=plt.figure(figsize=(8, 8))
  columns = 3
  rows = 3
  print(columns * rows, "samples from the dataset")
  i=1
  for a, b in dataset.take(columns * rows):
    print(a.numpy())
    print("------------")
    print(b.numpy())
    fig.add_subplot(rows, columns, i)
    plt.imshow(np.squeeze(a))
    plt.title("image shape:" + str(a.shape)+" ("+str(b.numpy()) + ")" +
              str(covert_onehot_string_labels(LABELS, b.numpy())))
    i = i+1
  plt.show()


show_samples(ds_test)


#buffer_size = ds_train_resize_scale.cardinality().numpy()/10
#ds_resize_scale_batched=ds_raw.repeat(3).shuffle(buffer_size=buffer_size).batch(64, )

ds_train_batched = ds_train.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)
ds_test_batched = ds_test.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)

print("Number of batches in train: ", ds_train_batched.cardinality().numpy())
print("Number of batches in test: ", ds_test_batched.cardinality().numpy())


# ---------------------------  Try & See ---------------------------------------
model.fit(ds_train_batched, validation_data=ds_test_batched, epochs=10)


# ------------------------------- Save A Model -----------------------------------

model.save("fire_smoke_vgg16_model.h5")
# ------------------------- Evaulate the model -----------------------------------
ds = ds_test_batched
print("Test Accuracy: ", model.evaluate(ds)[1])

# 10 sample predictions
ds = ds_test
predictions = model.predict(ds.batch(batch_size=10).take(1))
print("A sample output from the last layer (model) ", predictions[0])
y = []
print("10 Sample predictions:")
for (pred, (a, b)) in zip(predictions, ds.take(10)):
  pred[pred > 0.2] = 1
  pred[pred <= 0.2] = 0
  print("predicted: ", pred, str(covert_onehot_string_labels(LABELS, pred)),
        "Actual Label: (" + str(covert_onehot_string_labels(LABELS, b.numpy())) + ")")
  y.append(b.numpy())

# show_samples(pred)