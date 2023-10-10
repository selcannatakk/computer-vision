import keras
import tensorflow as tf
import os

from keras import layers, applications, activations, optimizers,losses, metrics


base_model = tf.keras.applications.EfficientNetB7(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None)


base_model.trainable = False
number_of_classes = 5
initializer = tf.initializers.GlorotUniform(seed=42)

# None  # tf.keras.activations.sigmoid or softmax
activation = activations.sigmoid

inputs = keras.Input(shape=(64, 64, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(number_of_classes, kernel_initializer=initializer, activation=activation)(x)

model = keras.Model(inputs, outputs)

# ------------------------------- Compile & Train ---------------------------------
model.compile(optimizer=optimizers.Adam(),
              loss=losses.BinaryCrossentropy(), # default from_logits=False
              metrics=[metrics.BinaryAccuracy()])