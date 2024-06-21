from torch.utils.data.dataset import Dataset
import os
import torch
import numpy as np
from tqdm import tqdm
import torchvision
from PIL import Image
from glob import glob


class CelebDataset(Dataset):
    def __init__(self, split, image_path, image_size=256, im_channels=3, ):
        self.split = split
        self.image_size = image_size
        self.im_channels = im_channels
        self.image_path = image_path
        self.images = self.load_images(self.image_path)


    def load_images(self, image_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(image_path), "images path {} does not exist".format(image_path)
        images = []
        fnames = glob(os.path.join(image_path, 'CelebA-HQ-img/*.{}'.format('png')))
        fnames += glob(os.path.join(image_path, 'CelebA-HQ-img/*.{}'.format('jpg')))
        fnames += glob(os.path.join(image_path, 'CelebA-HQ-img/*.{}'.format('jpeg')))

        for fname in tqdm(fnames):
            images.append(fname)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        image_tensor = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.image_size),
                torchvision.transforms.CenterCrop(self.image_size),
                torchvision.transforms.ToTensor(),
            ]
        )(img)
        img.close()

        image_tensor = (2*image_tensor)-1
        return image_tensor
