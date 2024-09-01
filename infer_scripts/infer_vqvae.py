import torch
import numpy as numpy
import os
import glob
from tqdm import tqdm
from models.vqvae import VQVAE
import yaml
import torchvision
from dataset.celeb_dataset import CelebDataset
from torchvision.utils import make_grid
import argparse
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    with open(args.config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create the dataset
    im_dataset_cls = {
        'celebhq': CelebDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])

    # This is only used for saving latents. Which as of now
    # is not done in batches hence batch size 1
    data_loader = DataLoader(im_dataset,
                             batch_size=1,
                             shuffle=False)

    num_images = train_config['num_samples']
    ngrid = train_config['num_grid_rows']




    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    main(args)
