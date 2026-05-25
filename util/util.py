"""Loss functions and helper utilities for SA-VSSM."""

import torch
import torch.nn.functional as F


def make_one_hot(labels, label_lengths, num_classes):
    """Convert label tensors to one-hot encoding."""
    batch_size, max_len = labels.shape
    one_hot = torch.zeros(batch_size, max_len, num_classes, dtype=torch.float32)
    for i in range(batch_size):
        for j in range(label_lengths[i]):
            one_hot[i, j, labels[i, j] - 1] = 1
    return one_hot


def loss_hinge_dis(dis_fake, dis_real, len_text_fake, len_text, mask_loss=True):
    """Hinge loss for discriminator."""
    mask_real = torch.ones_like(dis_real)
    mask_fake = torch.ones_like(dis_fake)

    if mask_loss and len(dis_fake.shape) > 2:
        for i in range(len(len_text)):
            mask_real[i, :, :, len_text[i]:] = 0
            mask_fake[i, :, :, len_text_fake[i]:] = 0

    loss_real = torch.sum(F.relu(1.0 - dis_real * mask_real)) / torch.sum(mask_real)
    loss_fake = torch.sum(F.relu(1.0 + dis_fake * mask_fake)) / torch.sum(mask_fake)
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake, len_text_fake, mask_loss=True):
    """Hinge loss for generator."""
    mask_fake = torch.ones_like(dis_fake)
    if mask_loss and len(dis_fake.shape) > 2:
        for i in range(len(len_text_fake)):
            mask_fake[i, :, :, len_text_fake[i]:] = 0
    return -torch.sum(dis_fake * mask_fake) / torch.sum(mask_fake)


def toggle_grad(model, requires_grad):
    """Enable or disable gradient computation for all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = requires_grad
