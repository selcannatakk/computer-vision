import keras
import tensorflow as tf
import os

from keras import layers, applications, activations, optimizers, losses, metrics


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

base_model = applications.vgg16.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(64, 64, 3),# VGG16 expects min 32 x 32
    include_top=False)  # Do not include the ImageNet classifier at the top.


base_model.trainable = False
number_of_classes = 2
initializer = tf.initializers.GlorotUniform(seed=42)

# None  # tf.keras.activations.sigmoid or softmax
activation = activations.sigmoid


inputs = keras.Input(shape=(64, 64, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(number_of_classes, kernel_initializer=initializer, activation=activation)(x)

model = keras.Model(inputs, outputs)
#
# inputs = keras.Input(shape=(64, 64, 3))
# x = base_model(inputs, training = False),
# model = keras.Sequential([
#     layers.GlobalAveragePooling2D()(x)
# ])
# initializer = tf.initializers.GlorotUniform(seed=42)
#
# activation = tf.keras.activations.sigmoid #None  # tf.keras.activations.sigmoid or softmax
#
# outputs = keras.layers.Dense(number_of_classes,
#                              kernel_initializer=initializer,
#                              activation=activation)(x)


model.save("fire_smoke_vgg16_model")
# ------------------------------- Compile & Train ---------------------------------
model.compile(optimizer=optimizers.Adam(),
              loss=losses.BinaryCrossentropy(), # default from_logits=False
              metrics=[metrics.BinaryAccuracy()])
# --------------------------------- Model a save -----------------------------------
# model.save("model.keras")
# Model.save_weights(filepath, overwrite=True, save_format=None, options=None)
# loaded_model = tf.keras.models.load_model("model.keras")
# ------------------------------------------------------------------------------------------------------------------------
# Softmax
#     softmax tahmin edilen tüm değerleri bir olasılık dağılımı olarak normalleştirdiğinden, tahmin olarak yalnızca tek bir etiket seçebilir .
#     Sadece bir etiket 0,5'ten daha yüksek değer alabilir.
#     Bu nedenle, softmax en fazla yalnızca TEK bir sınıfı tahmin edebilir!

# Sigmoid
#     Sigmoid, son katmanın aktivasyon fonksiyonu olarak uygulandığında, sigmoid tahmin edilen her logit değerini bağımsız olarak
#     0 ile 1 arasında normalleştirdiğinden tahmin olarak birden fazla etiket seçebilir .
#
#keras.metrics.BinaryAccuracy()
#    Tahminlerin ikili etiketlerle ne sıklıkta eşleştiğini hesapladığından doğruluğu ölçmek için keras.metrics.BinaryAccuracy() kullanmamız gerekiyor .
#    her bir tahmin öğesini gerçek etiketlerin karşılık gelen öğesiyle karşılaştırmamız gerekir .

