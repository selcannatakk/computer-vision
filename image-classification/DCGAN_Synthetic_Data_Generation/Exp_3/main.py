import os
import cv2
import torchvision.transforms as T
import jovian
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from visualization import save_samples
from gan_model import discriminator, generator
from gan_model_training import fit
from IPython.display import Image


DATA_DIR = '../data/face/'
print(os.listdir(DATA_DIR+'images')[:10])

image_size = 64 #pixel değeri
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)#standart sapma:0,5

train_ds = ImageFolder(DATA_DIR, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)]))

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

# show_batch(train_dl, stats)

# -------------------------- DEVICE CPU -----------------------------

#
# def get_default_device():
#     """Pick GPU if available, else CPU"""
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu')
#
#
# def to_device(data, device):
#     """Move tensor(s) to chosen device"""
#     if isinstance(data, (list, tuple)):
#         return [to_device(x, device) for x in data]
#     return data.to(device, non_blocking=True)
#
#
# class DeviceDataLoader():
#     """Wrap a dataloader to move data to a device"""
#
#     def __init__(self, dl, device):
#         self.dl = dl
#         self.device = device
#
#     def __iter__(self):
#         """Yield a batch of data after moving it to device"""
#         for b in self.dl:
#             yield to_device(b, self.device)
#
#     def __len__(self):
#         """Number of batches"""
#         return len(self.dl)
#
#
# device = get_default_device()
# verilerimizi cpu ya taşıyoruz
# train_dl = DeviceDataLoader(train_dl, device)
# Model oluşturduktan sonra discriminator modelimizi cpu ya tasıyoruz.
# discriminator = to_device(discriminator, device)

#random input
latent_size = 128
noise = torch.randn(batch_size, latent_size, 1, 1) # random  tensors
fake_images = generator(noise)

print(fake_images.shape)
# show_images(fake_images)

# Model oluşturduktan sonra generator modelimizi cpu ya tasıyoruz.
# generator = to_device(generator, device)

sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)
# Modeli eğitirken bireysel olarak oluşturulan görüntülerin zaman içinde nasıl geliştiğini
# görmek için üreticiye sabit bir giriş vektörleri seti kullanacağız
fixed_latent = torch.randn(64, latent_size, 1, 1)
save_samples(0, fixed_latent, generator, sample_dir, stats)

#------------------------------- TRAIN A MODEL --------------------------------
lr = 0.0002
epochs = 25
# jovian.reset()
# jovian.log_hyperparams(lr=lr, epochs=epochs)

history = fit(epochs, lr, fixed_latent, train_dl)

losses_g, losses_d, real_scores, fake_scores = history

# jovian.log_metrics(loss_g=losses_g[-1],
#                    loss_d=losses_d[-1],
#                    real_score=real_scores[-1],
#                    fake_score=fake_scores[-1])

# Save the model checkpoints
torch.save(generator.state_dict(), 'G.pth')
torch.save(discriminator.state_dict(), 'D.pth')

Image('./generated/generated-images-0001.png')

vid_fname = 'gans_training.avi'

files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'generated' in f]
files.sort()

out = cv2.VideoWriter(vid_fname,cv2.VideoWriter_fourcc(*'MP4V'), 1, (530,530))
[out.write(cv2.imread(fname)) for fname in files]
out.release()