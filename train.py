from pkg_resources import require
from sklearn import discriminant_analysis
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from utils import *
from dataset import get_data_loader

from models.discriminator import Discriminator
from models.generator import Generator

import wandb

# Sample data from the loader
def get_sample(loader):
    while True:
        for batch in loader:
            yield batch[0]


# Train Generator and Discriminator
def train(loader, generator, discriminator, g_optim, d_optim, device, args):
    '''
    Training function of Generator and Discriminator. 
    
    Arguments:
        args:          Contains process information 
        loader:        Data Loader
        generator:     Style-Swin Transformer Generator
        discriminator: Conv-Based discriminator
        g_optim:       Generator Optimizer
        d_optim:       Discriminator Optimizer
        device:        Training device
        
    '''
    # Yield a batch of data
    loader = get_sample(loader)

    # Set the configuration of training
    losses = {}
    # Initialize gradient penalty
    grad_pen_loss = torch.tensor(0.0, device=device)
    # L2 loss 
    l2 = nn.MSELoss()
    # Gradient clipping
    gradient_clip = nn.utils.clip_grad_value_

    for iters in range(args.n_iters):

        # ------------------ Train Discriminator -------------------- #
        generator.train()
        # Get batch of images and put them to device
        real_img = next(loader).to(device)
        # Avoid Generator to be updated
        adjust_gradient(generator, False)
        # Permit only discriminator to be updated
        adjust_gradient(discriminator, True)
        
        # Sample random noise from normal distribution
        noise_dim = (args.batch_size, args.style_dim) # ~ initial channel 512
        noise = torch.randn(noise_dim).to(device) # ~ maybe .cuda()?

        # Generate Fake image from random noise
        fake_img = generator(noise)
        # Get discriminator performance on generated images
        fake_pred = discriminator(fake_img)
        # Get discriminator performance on real images
        real_pred = discriminator(real_img)

        # Calculate Discriminator Loss
        d_loss = discriminator_loss(real_pred, fake_pred)

        # Update discriminator
        discriminator.zero_grad()
        d_loss.backward()
        # gradient_clip(discriminator.parameters(), 5.0)
        d_optim.step()

        # # Employ Gradient Penalty
        # if iters % args.d_reg_every == 0:
        #     real_img.requires_grad = True
        #     # Get the prediction on updated discriminator
        #     real_pred = discriminator(real_img)
        #     # Calculate the R1 loss: Gradient penalty
        #     #grad_pen_loss = gradient_penalty(real_pred, real_img)
            
        #     # Update Discriminator
        #     discriminator.zero_grad()
        #     (args.scaler * (args.r1 / 2 * grad_pen_loss * args.d_reg_every)).backward()
        #     grad_pen_loss.backward()  # ~ Ideally add some weighting... to this loss
        #     d_optim.step()

        # Save the losses
        losses['discriminator'] = d_loss        
        losses['gradient_penalty'] = grad_pen_loss

        
        # ------------------ Train Generator -------------------- #
        
        # Avoid Discriminator to be updated
        adjust_gradient(discriminator, False)
        # Permit only generator to be updated
        adjust_gradient(generator, True)

        # Get the next batch of real images
        real_img = next(loader).to(device)

        # Sample random noise from normal distribution
        noise_dim = (args.batch_size, args.style_dim) # ~ initial channel 512
        noise = torch.randn(noise_dim).to(device) # ~ maybe .cuda()?

        # Generate Fake image from random noise
        fake_img = generator(noise)
        # Get discriminator performance on generated images
        fake_pred = discriminator(fake_img)
        
        # Calculate the Generator loss
        g_loss = generator_loss(fake_pred) # Ideally, add weight

        # Save the loss
        losses['generator'] = g_loss

        # Update Generator
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # Log and Save
        if iters % args.print_freq == 0:
            visualize_loss = {
                'd_loss': d_loss,
                'g_loss': g_loss,
                'grad_pen_loss': grad_pen_loss,
            }
            
            wandb.log(visualize_loss, step=iters)
            # print('Iters: {iters}\t ')

            torch.save({'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'args': args}, 'checkpoints/state_dict')
        


if __name__ == '__main__':
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'     
    # Parse Arguments
    args = parse_arguments()
    print(args.resolution)
    # WandB
    wandb.init(project="styleswin", entity="metugan")
    
    # Generator
    generator = Generator(
                    dim=args.dim,
                    style_dim=args.style_dim,
                    n_style_layers=args.n_style_layers,
                    n_heads=args.n_heads,
                    resolution=args.resolution,
                    attn_drop=args.attn_drop 
                ).to(device)

    generator_learning_rate = args.gen_lr
    generator_betas = (args.beta1 , args.beta2)
    g_optim = optim.Adam(generator.parameters(), 
                        lr=generator_learning_rate, 
                        betas=generator_betas)

    # Discriminator
    discriminator = Discriminator(
                        n_activ_maps=args.n_activ_maps,
                        n_channels=3,
                        resolution=args.resolution
                    ).to(device)

    discriminator_learning_rate = args.disc_lr
    discriminator_betas = (args.beta1 , args.beta2)
    d_optim = optim.Adam(discriminator.parameters(), 
                        lr=discriminator_learning_rate, 
                        betas=discriminator_betas)
    
    # Get DataLoader
    datasetname = 'LSUN'
    root = 'data/'
    batch_size = args.batch_size
    loader = get_data_loader(datasetname, root, batch_size)

    train(loader=loader, generator=generator, discriminator=discriminator,
        g_optim=g_optim, d_optim=d_optim, device=device, args=args)