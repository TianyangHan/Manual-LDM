import torch
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model):

    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    xt = torch.randn((1,autoencoder_model_config['z_channels']),
                    im_size, im_size
                    ).to(device)

    text_prompt = ['She is a woman with blond hair. She is wearing lipstick.']
    neg_prompts = ['He is a man.']
    empty_prompt = ['']

    text_prompt_embed = get_text_representation(text_prompt, text_tokenizer, text_model, device)
    empty_text_embed = get_text_representation(empty_prompt, text_tokenizer, text_model, device)
    assert empty_text_embed.shape == text_prompt_embed.shape

    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    dataset = CelebDataset(split='train', im_path=dataset_config['im_path'],
                            im_size=dataset_config['im_size'], im_channels=dataset_config['im_channels'],
                            use_latents=True, latent_path=os.path.join(train_config['task_name'], train_config['vqvaw_latent_dir_name']),
                            condition_config=condition_config)
    mask_idx = random.randint(0,len(dataset.masks))
    mask = dataset.get_mask(mask_idx).unsqueeze(0).to(device)

    uncond_input = {
        'text':empty_text_embed,
        'image':torch.zeros_like(mask)
    }

    cond_input = {
        'text':text_prompt_embed,
        'image':mask
    }
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        t = (torch.ones((xt.shape[0],))*i).long().to(device)
        noise_pred_cond = model(xt,t,cond_input)
        # concat uncond and cond at dim=0, than torch.chunk
        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, t,unconde_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond

        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        if i==0:
            # decode image
            ims = vae.decode(xt)
        else:
            ims = x0_pred

        # set element range [-1,1]
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        tms = (ims+1)/2
        grid = make_grid(ims, nrow=10)
        img = torchvision.transforms.ToPILImage()(grid)
        
        if not os.path.exists(os.path.join(train_config['task_name'], 'cond_text_image_samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'cond_text_image_samples'))
        img.save(os.path.join(train_config['task_name'], 'cond_text_image_samples', 'x0_{}.png'.format(i)))
        img.close()