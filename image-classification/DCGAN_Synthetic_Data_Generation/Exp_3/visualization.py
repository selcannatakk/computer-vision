import os
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from torchvision.utils import save_image


# images denormalize etme:
def denorm(img_tensors, stats):
    return img_tensors * stats[1][0] + stats[0][0]


def show_images(images, stats, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax], stats), nrow=8).permute(1, 2, 0))


def show_batch(dl, stats, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


def save_samples(index, latent_tensors, generator, sample_dir, stats, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images, stats), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    # if show:
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

