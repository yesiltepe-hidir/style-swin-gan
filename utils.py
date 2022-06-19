from ast import arg
from xmlrpc.client import getparser
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse

'''
dim = 4
batch_size = 2
style_dim = 256
n_mlp = 8
channel_dim = 512
attn_drop = 0.
n_heads = 16
resolution = 256

# style-massage
n_style_layers = 8
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description= 'Re-implementation of the Style-Swin Paper')
    parser.add_argument('--batch_size',  type=int, default=2)
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--dim', type=int, default=4, help= 'Initial constant input dimension: Height and Width')
    parser.add_argument('--channel_dim', type=int, default=64)
    parser.add_argument('--style_dim', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--n_style_layers', type=int, default=8)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--print_freq', type=int, default=10)
    
    # disc -args
    parser.add_argument('--n_activ_maps', type=int, default=32)

    # optim - args
    parser.add_argument("--scaler", type=float, default=1)
    parser.add_argument("--gen_lr", type=float, default=5e-2)
    parser.add_argument("--disc_lr", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--attn_drop", type=float, default=0)

    args = parser.parse_args(args=[])

    return args

def discriminator_loss(real_pred, fake_pred):
    """
    Softplus Loss will be used for Discriminator Logistic Loss:
        >> Softplus(x) = ln(1 + exp(x))
    """
    fake_loss = F.softplus(fake_pred)
    real_loss = F.softplus(-real_pred)

    return fake_loss.mean() + real_loss.mean()


def generator_loss(fake_pred):
    """
    Generator Loss is the same function as Discriminator loss: Softplus
    """
    gen_loss = F.softplus(-fake_pred).mean()
    return gen_loss


def adjust_gradient(model, req_grad=True):
    """
    Adjusts the gradient changes required for training Discriminator
    and Generator.
    """
    # Change the model parameters `requires_grad` parameter to input flag
    for parameter in model.parameters():
        parameter.requres_grad = req_grad


def gradient_penalty(real_pred, real_img):
    """
    Also called R1 Loss. It is used in Discriminator as a regularization.
    Takes Real images and prediction for real images.
    Returns the sum of squared gradients.
    """
    outputs = real_pred.sum()
    (grads,) = torch.autograd(outputs=outputs, inputs=real_img, create_graph=True)
    penalty = grads.pow(2).reshape(grads.size(0), -1).sum(1).mean()

    return penalty
