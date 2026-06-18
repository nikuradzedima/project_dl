import math

import lpips
import torch
from torch.nn import functional as F
from torchmetrics.functional import structural_similarity_index_measure


class MSEMetric:
    name = "MSE"

    def __call__(self, lensed_roi, reconstruction_roi):
        return float(F.mse_loss(reconstruction_roi, lensed_roi).detach().cpu().item())


class PSNRMetric:
    name = "PSNR"

    def __call__(self, lensed_roi, reconstruction_roi):
        mse = F.mse_loss(reconstruction_roi, lensed_roi).detach().cpu().item()
        return float(10.0 * math.log10(1.0 / max(mse, 1e-12)))


class SSIMMetric:
    name = "SSIM"

    def __call__(self, lensed_roi, reconstruction_roi):
        value = structural_similarity_index_measure(
            reconstruction_roi.clamp(0, 1),
            lensed_roi.clamp(0, 1),
            data_range=1.0,
        )
        return float(value.detach().cpu().item())


class LPIPSMetric:
    name = "LPIPS"

    def __init__(self, net="vgg", device="auto"):
        self.net = net
        self.device = device
        self.metric = None

    def __call__(self, lensed_roi, reconstruction_roi):
        if self.metric is None:
            if self.device == "auto":
                device = reconstruction_roi.device
            else:
                device = torch.device(self.device)
            self.metric = lpips.LPIPS(net=self.net).to(device).eval()
            for parameter in self.metric.parameters():
                parameter.requires_grad_(False)
        value = self.metric(
            reconstruction_roi.clamp(0, 1) * 2 - 1,
            lensed_roi.clamp(0, 1) * 2 - 1,
        ).mean()
        return float(value.detach().cpu().item())
