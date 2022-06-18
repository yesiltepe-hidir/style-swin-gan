import torch
import torch.nn as nn
from torch.nn import functional as F


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
