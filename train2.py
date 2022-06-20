# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import builtins
import os
import sys
from datetime import timedelta
from tqdm import tqdm

import torch
import torchvision
import torchvision.datasets as datasets
from torch import autograd, nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim, real

from utils import *
from dataset import get_data_loader

from models.discriminator import Discriminator
from models.generator import Generator

import wandb
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

import time

from models.discriminator import Discriminator
from models.generator import Generator




def requires_grad(model, flag=True):
    acc_grad = 0
    for p in model.parameters():
        p.requires_grad = flag
        if p.grad is not None:
            acc_grad += (p.grad.clone()).mean()
    print(acc_grad)

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    assert type(real_pred) == type(fake_pred), "real_pred must be the same type as fake_pred"
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def tensor_transform_reverse(image):
    assert image.dim() == 4
    moco_input = torch.zeros(image.size()).type_as(image)
    moco_input[:,0,:,:] = image[:,0,:,:] * 0.229 + 0.485
    moco_input[:,1,:,:] = image[:,1,:,:] * 0.224 + 0.456
    moco_input[:,2,:,:] = image[:,2,:,:] * 0.225 + 0.406
    return moco_input

def train(loader, generator, discriminator, g_optim, d_optim, device, args):
    
    loader = sample_data(loader)

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    accum = 0.5 ** (32 / (10 * 1000))
    loss_dict = {}
    l2_loss = torch.nn.MSELoss()
    loss_dict = {}

    g_module = generator
    d_module = discriminator

    print(" -- start training -- ")
    end = time.time()

    for idx in range(args.n_iters):
        i = idx + args.start_iter
        print(f'iter: {i}')
        if i > args.n_iters:
            print("Done!")
            break

        # Train D
        generator.train()

        this_data = next(loader)
        real_img = this_data[0]
        real_img = real_img.to(device)
        
        
        

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        noise = torch.randn((args.batch, args.style_dim)).cuda()

        fake_img = generator(noise)
        fake_img_cpy = fake_img.clone()
        # plt.imshow(fake_img_cpy[0].permute(1,2,0).cpu().detach())
        # plt.show()

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)*args.gan_weight
        
        loss_dict["d"] = d_loss

        discriminator.zero_grad()
        d_loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True

            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.gan_weight * (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0])).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # Train G
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        this_data = next(loader)
        real_img = this_data[0]
        real_img = real_img.to(device)

        noise = torch.randn((args.batch, args.style_dim)).cuda()
        fake_img = generator(noise)
        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)*args.gan_weight

        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()


        # Log and Save
        if i % args.print_freq == 0:
            visualize_loss = {
                'd_loss': d_loss/args.gan_weight,
                'g_loss': g_loss/args.gan_weight,
                'grad_pen_loss': r1_loss/args.gan_weight,
            }
            
            wandb.log(visualize_loss, step=i)
            # print('Iters: {iters}\t ')

            torch.save({'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'args': args}, 'checkpoints/state_dict')
        



if __name__ == "__main__":
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'     
    # Parse Arguments
    args = parse_arguments()

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
    batch_size = args.batch
    loader = get_data_loader(datasetname, root, batch_size)

    print("-" * 80)
    print("Generator: ")
    print(generator)
    print("-" * 80)
    print("Discriminator: ")
    print(discriminator)

    train(loader=loader, generator=generator, discriminator=discriminator,
        g_optim=g_optim, d_optim=d_optim, device=device, args=args)



    
    