import math

import torch
from torch import nn
from torch.nn import functional as F


def inverse_softplus(value):
    return torch.log(torch.expm1(value))


def center_pad(x, size):
    height, width = x.shape[-2:]
    target_height, target_width = size
    pad_height = target_height - height
    pad_width = target_width - width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    return F.pad(x, (left, right, top, bottom))


def center_crop(x, size):
    height, width = x.shape[-2:]
    target_height, target_width = size
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return x[..., top : top + target_height, left : left + target_width]


def soft_threshold(x, threshold):
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


def circular_gradient(x):
    grad_y = torch.roll(x, shifts=-1, dims=-2) - x
    grad_x = torch.roll(x, shifts=-1, dims=-1) - x
    return torch.stack((grad_y, grad_x), dim=2)


def circular_divergence(gradient):
    grad_y = gradient[:, :, 0]
    grad_x = gradient[:, :, 1]
    div_y = torch.roll(grad_y, shifts=1, dims=-2) - grad_y
    div_x = torch.roll(grad_x, shifts=1, dims=-1) - grad_x
    return div_y + div_x


def conv_forward(x, kernel_fft):
    return torch.fft.ifft2(torch.fft.fft2(x) * kernel_fft).real


def conv_adjoint(x, kernel_fft):
    return torch.fft.ifft2(torch.fft.fft2(x) * kernel_fft.conj()).real


class UnrolledADMM(nn.Module):
    def __init__(
        self,
        num_iterations,
        learnable=True,
        init_mu=1e-4,
        init_tau=2e-4,
        pad_factor=2.0,
        fft_multiple=16,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.learnable = learnable
        self.pad_factor = pad_factor
        self.fft_multiple = fft_multiple
        mu = torch.full((num_iterations,), float(init_mu))
        tau = torch.full((num_iterations,), float(init_tau))
        if learnable:
            self.raw_mu1 = nn.Parameter(inverse_softplus(mu))
            self.raw_mu2 = nn.Parameter(inverse_softplus(mu))
            self.raw_mu3 = nn.Parameter(inverse_softplus(mu))
            self.raw_tau = nn.Parameter(inverse_softplus(tau))
        else:
            self.register_buffer("mu1", mu)
            self.register_buffer("mu2", mu)
            self.register_buffer("mu3", mu)
            self.register_buffer("tau", tau)

    def forward(self, measurement, psf):
        image_size = measurement.shape[-2:]
        padded_size = self.padded_size(image_size)
        b = center_pad(measurement, padded_size)
        kernel_fft = self.prepare_kernel(psf, padded_size)
        crop_mask = self.crop_mask(measurement, padded_size)
        diff_power = self.diff_power(
            padded_size, measurement.device, measurement.dtype
        )
        x = torch.zeros_like(b)
        v = torch.zeros_like(b)
        w = torch.zeros_like(b)
        u = torch.zeros(
            b.shape[0], b.shape[1], 2, b.shape[2], b.shape[3], device=b.device
        )
        alpha1 = torch.zeros_like(b)
        alpha2 = torch.zeros_like(u)
        alpha3 = torch.zeros_like(b)
        h_power = kernel_fft.abs().square()
        for idx in range(self.num_iterations):
            mu1, mu2, mu3, tau = self.parameters_at(idx)
            hx = conv_forward(x, kernel_fft)
            v = (alpha1 + mu1 * hx + b) / (crop_mask + mu1)
            u = soft_threshold(circular_gradient(x) + alpha2 / mu2, tau / mu2)
            w = torch.relu(alpha3 / mu3 + x)
            rhs = (
                mu3 * w
                - alpha3
                + circular_divergence(mu2 * u - alpha2)
                + conv_adjoint(mu1 * v - alpha1, kernel_fft)
            )
            denominator = mu1 * h_power + mu2 * diff_power + mu3
            x = torch.fft.ifft2(torch.fft.fft2(rhs) / denominator).real
            alpha1 = alpha1 + mu1 * (conv_forward(x, kernel_fft) - v)
            alpha2 = alpha2 + mu2 * (circular_gradient(x) - u)
            alpha3 = alpha3 + mu3 * (x - w)
        return center_crop(x, image_size)

    def parameters_at(self, idx):
        if self.learnable:
            mu1 = F.softplus(self.raw_mu1[idx]) + 1e-8
            mu2 = F.softplus(self.raw_mu2[idx]) + 1e-8
            mu3 = F.softplus(self.raw_mu3[idx]) + 1e-8
            tau = F.softplus(self.raw_tau[idx]) + 1e-8
        else:
            mu1 = self.mu1[idx]
            mu2 = self.mu2[idx]
            mu3 = self.mu3[idx]
            tau = self.tau[idx]
        return mu1, mu2, mu3, tau

    def padded_size(self, image_size):
        height, width = image_size
        padded_height = self.round_fft_size(int(math.ceil(height * self.pad_factor)))
        padded_width = self.round_fft_size(int(math.ceil(width * self.pad_factor)))
        return padded_height, padded_width

    def round_fft_size(self, value):
        multiple = self.fft_multiple
        return int(math.ceil(value / multiple) * multiple)

    def prepare_kernel(
        self, psf, padded_size
    ):
        kernel = center_pad(psf, padded_size)
        kernel = torch.fft.ifftshift(kernel, dim=(-2, -1))
        return torch.fft.fft2(kernel)

    def crop_mask(
        self, measurement, padded_size
    ):
        mask = torch.ones(
            measurement.shape[0],
            measurement.shape[1],
            measurement.shape[-2],
            measurement.shape[-1],
            device=measurement.device,
            dtype=measurement.dtype,
        )
        return center_pad(mask, padded_size)

    def diff_power(
        self, padded_size, device, dtype
    ):
        height, width = padded_size
        kernel_y = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
        kernel_x = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
        kernel_y[..., 0, 0] = -1
        kernel_y[..., 1 % height, 0] = 1
        kernel_x[..., 0, 0] = -1
        kernel_x[..., 0, 1 % width] = 1
        fft_y = torch.fft.fft2(kernel_y)
        fft_x = torch.fft.fft2(kernel_x)
        return fft_y.abs().square() + fft_x.abs().square()
