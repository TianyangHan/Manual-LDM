import torch
import argparse
import numpy as np
from torch.utils.data.dataloader import DataLoader
from models.vqvae import VQVAE
from model.discriminator import Discriminator



def train(args):

    device = torch.device('cuda' if torch.cuda.is_avaiable() else 'cpu')

    data_config = config['dataset_params']
    auto_encoder_config = config['auto_encoder_params']
    train_config = config['train_params']




    

    dataset_cls = {
        'mnist':MinistDataset,
        'celebhq':CelebDataset
    }.get(data_config['name'])
    dataset = dataset_cls()
    data_loader = DataLoader(
        dataset, batch_size=train_config['autoencoder_batch_size'],
        shuffle=True,
    )


    model = VQVAE(in_channels=data_config['in_channels'], model_config = autoencoder_config).to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)

    l_loss = troch.nn.MSELoss()
    disc_loss = torch.nn.MSELoss()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)