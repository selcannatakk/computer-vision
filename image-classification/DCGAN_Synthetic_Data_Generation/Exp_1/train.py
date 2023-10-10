import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import os
import tensorflow.keras as tk

# Bring in the sequential api for the generator and discriminator
from tensorflow.keras.models import Sequential
# Bring in the layers for the neural network
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
from tensorflow import keras
from tensorflow.python.client import device_lib
from visualization import show, train_dcgan, generate_and_save_images
from PIL import Image

tf.test.gpu_device_name()
device_lib.list_local_devices()

print('Tensorflow version:', tf.__version__)

#2.ADIM: Verilerin yüklenmesi ve önişlemlerin gerçekleştirilmesi
#Fashion MNIST veri kümesinin keras yoluyla indiriyoruz.

(x_train, y_train), (x_test, y_test) = tk.datasets.fashion_mnist.load_data()
# ölçeklendirme
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Veri kümesinden 10x10 piksel büyüklüklü 25 tane örnek ekrana yazdırıp neye benzediğine bakalım.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()

#3.ADIM: Eğtim verileri ve gruplarının oluşturulması

# # Reload the dataset
# ds = tfds.load('fashion_mnist', split='train')
# # Running the dataset through the scale_images preprocessing step
# ds = ds.map(scale_images)
# # Cache the dataset for that batch
# ds = ds.cache()
# # Shuffle it up
# ds = ds.shuffle(60000)
# # Batch into 128 images per sample
# ds = ds.batch(128)
# # Reduces the likelihood of bottlenecking
# ds = ds.prefetch(64)

# ds.as_numpy_iterator().next().shape

batch_size = 32
# Bu veri kümesi, bir ara belleği buffer_size elemanları ile doldurur,
#ardından seçilen elemanları yeni elemanlarla değiştirerek rastgele örnekler.
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
# Bu veri kümesinin ardışık öğelerini toplu olarak birleştirir.
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1) #?

#4.ADIM: DCGAN için üretici (generator) ağının oluşturulması
# ÜRETİCİ KATMANINDAKİ EVRİŞİMLİ SİNİR AĞI
num_features = 100 # öznitelik sayısı

# giriş değerini verdiğimiz features sayısına göre başlatıyoruz
# Conv2DTranpose versiyonunu kullanıyoruz.
def build_generator():
    model = Sequential()

    # Takes in random values and reshapes it to 7x7x128
    # Beginnings of a generated image
    model.add(Dense(7 * 7 * 128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))

    # Upsampling block 1
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    # Upsampling block 2
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    # Convolutional block 1
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Convolutional block 2
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Conv layer to get to one channel
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

    return model

generator = build_generator()

generator.summary()

# noise = başlangıctaki random input (gürültü)
noise = tf.random.normal(shape=[1, num_features])
generated_images = generator(noise, training=False)
show(generated_images, 1)

#5.ADIM: DCGAN için ayırıcı (discriminator) ağının oluşturulması
def build_discriminator():
    model = Sequential()

    # First Conv Block
    model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Second Conv Block
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Third Conv Block
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Fourth Conv Block
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Flatten then pass to dense layer
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model

discriminator = build_discriminator()
discriminator.summary()

# Üretilen görsel için ayırt edici %50nin altında bir değer ürettir ilk adım için
decision = discriminator(generated_images)
print(decision)

# ------------------------GAN MODELİ------------------------
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan = keras.models.Sequential([generator, discriminator])
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

#7.ADIM: Eğitim işlemlerini görseleştirilmesi
seed = tf.random.normal(shape=[batch_size, 100])

#8.ADIM: DCGAN'ın eğitilmesi
# Eğitim için yeniden boyutlandırmanın yapılması
# rcb görüntüyü nasıl verebiliriz?
x_train_dcgan = x_train.reshape(-1, 28, 28, 1) * 2. - 1.

#Batch size boyutunun ve shuffle özelliklerinin belirlenmesi
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(x_train_dcgan)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

epochs = 3
# Modeli train etme
train_dcgan(gan, dataset, batch_size, num_features, epochs=epochs)

#9.ADIM: DCGAN ile sentetik görüntülerin oluşturulması
noise = tf.random.normal(shape=[batch_size, num_features])
generated_images = generator(noise)
show(generated_images, 8)

# Sonuçları GIF olarak göstermek için bu kısmı çalıştırın.
# Kaynak: https://www.tensorflow.org/tutorials/generative/dcgan#create_a_gif
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

Image(open(anim_file,'rb').read())

# generator.save('face_generator.h5')
# discriminator.save('face_discriminator.h5')

