import torch
import argparse
import numpy as np
from torch.utils.data.dataloader import DataLoader
from models.vqvae import VQVAE
from model.discriminator import Discriminator
from model.lpips import LPIPS
from tqdm import tqdm


def train(args):

    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    device = torch.device('cuda' if torch.cuda.is_avaiable() else 'cpu')

    data_config = config['dataset_params']
    auto_encoder_config = config['auto_encoder_params']
    train_config = config['train_params']

    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################

    dataset_cls = {
        'mnist':MinistDataset,
        'celebhq':CelebDataset
    }.get(data_config['name'])
    dataset = dataset_cls()
    data_loader = DataLoader(
        dataset, batch_size=train_config['autoencoder_batch_size'],
        shuffle=True,
    )

    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)



    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    num_epochs = train_config['autoencoder_epochs']


    model = VQVAE(in_channels=data_config['in_channels'], model_config = autoencoder_config).to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    lpips_model = LPIPS().eval().to(device)


    recon_loss = troch.nn.MSELoss()
    disc_loss = torch.nn.MSELoss()

    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))


    discriminator_step_start = train_config['disc_start']
    step = 0

    accumulate_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0


    for epoch_idc in range(num_epochs):
        reconstruction_losses = []
        code_boook_losses = []
        perceptual_losseses = []
        discriminator_losses = []
        generator_losses = []
        losses = []

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        for index, data in tqdm(enumerate(data_loader)):
            step += 1
            data = data.float().to(device)

            model_output = model(data)
            output, z, quantize_loss = model_output

            # Image Saving Logic
            if step % image_save_steps == 0 or step == 1:
                sample_size = min(8, data.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((data[:sample_size] + 1) / 2).detach().cpu()
                
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                if not os.path.exists(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples')):
                    os.mkdir(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples'))
                img.save(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples',
                                      'current_autoencoder_sample_{}.png'.format(img_save_count)))
                img_save_count += 1
                img.close()



            # generator loss
            #L2 loss
            reconstruction_loss = recon_loss(output, data)
            reconstruction_losses.append(reconstruction_loss.item())
            reconstruction_loss = reconstruction_loss / accumulate_steps
            generator_loss = (
                reconstruction_loss + (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / accumulate_steps) +
                      (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / accumulate_steps)
            )
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())

            # adversial
            if step > discriminator_step_start:
                discriminator_fake_pred = discriminator(output)
                discriminator_fake_loss = disc_loss(discriminator_fake_pred, torch.ones(discriminator_fake_pred.shape,
                                                    device = discriminator_fake_pred.device))
                generator_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                generator_loss += train_config['disc_weight'] * disc_fake_loss / accumulate_steps

            # lpips
            lpips_loss = torch.mean(lpips_model(output, data)) / accumulate_steps
            perceptual_losseses.append(train_config['perceptual_weight'] * lpips_loss.item())
            generator_loss += train_config['perceptual_weight'] * lpips_loss / accumulate_steps
            losses.append(generator_loss.item())
            generator_loss.backward()


            # discriminator loss
            if step > discriminator_step_start:
                fake = output
                discriminator_fake_pred = discriminator(fake.detach())
                discriminator_real_loss = discriminator(data)
                discriminator_fake_loss = disc_loss(discriminator_fake_pred, torch.zeros(discriminator_fake_pred.shape,
                                                    device = discriminator_fake_pred.device))
                discriminator_real_loss = disc_loss(discriminator_fake_pred, torch.ones(discriminator_real_pred.shape,
                                                    device = discriminator_real_pred.device))

                discriminator_loss = train_config['disc_weight'] * (discriminator_fake_loss + discriminator_real_loss) / 2
                discriminator_losses.append(discriminator_loss.item())
                discriminator_loss /= accumulate_steps
                discriminator_loss.backward()

                if step % accumulate_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            if step % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        if len(disc_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(recon_losses),
                       np.mean(perceptual_losses),
                       np.mean(codebook_losses),
                       np.mean(gen_losses),
                       np.mean(disc_losses)))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
                  format(epoch_idx + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses),
                         np.mean(codebook_losses)))
        
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']))
        torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                            train_config['vqvae_discriminator_ckpt_name']))
    print('Done Training...')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)