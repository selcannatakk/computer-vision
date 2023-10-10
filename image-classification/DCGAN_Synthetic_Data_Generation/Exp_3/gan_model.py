import torch.nn as model

discriminator = model.Sequential(
    # in: 3 x 64 x 64

    model.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    model.BatchNorm2d(64),
    model.LeakyReLU(0.2, inplace=True),
    # out: 64 x 32 x 32

    model.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    model.BatchNorm2d(128),
    model.LeakyReLU(0.2, inplace=True),
    # out: 128 x 16 x 16

    model.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    model.BatchNorm2d(256),
    model.LeakyReLU(0.2, inplace=True),
    # out: 256 x 8 x 8

    model.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    model.BatchNorm2d(512),
    model.LeakyReLU(0.2, inplace=True),
    # out: 512 x 4 x 4

    model.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    model.Flatten(),
    model.Sigmoid())

latent_size = 128
generator = model.Sequential(
    # in: latent_size x 1 x 1

    model.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    model.BatchNorm2d(512),
    model.ReLU(True),
    # out: 512 x 4 x 4

    model.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    model.BatchNorm2d(256),
    model.ReLU(True),
    # out: 256 x 8 x 8

    model.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    model.BatchNorm2d(128),
    model.ReLU(True),
    # out: 128 x 16 x 16

    model.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    model.BatchNorm2d(64),
    model.ReLU(True),
    # out: 64 x 32 x 32

    model.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    model.Tanh()
    # out: 3 x 64 x 64
)

