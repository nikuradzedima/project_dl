import lpips
from torch import nn
from torch.nn import functional as F


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        mse_weight=1.0,
        lpips_weight=0.02,
        auxiliary_weight=0.1,
        lpips_net="vgg",
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.auxiliary_weight = auxiliary_weight
        if lpips_weight > 0:
            self.lpips = lpips.LPIPS(net=lpips_net)
            for parameter in self.lpips.parameters():
                parameter.requires_grad_(False)
        else:
            self.lpips = None

    def forward(
        self,
        lensed_roi,
        reconstruction_roi,
        inverted_roi,
    ):
        mse_loss = F.mse_loss(reconstruction_roi, lensed_roi)
        loss = self.mse_weight * mse_loss
        output = {"loss": loss, "mse_loss": mse_loss}
        if self.lpips is not None:
            lpips_loss = self.lpips(
                self.to_lpips_range(reconstruction_roi),
                self.to_lpips_range(lensed_roi),
            ).mean()
            output["lpips_loss"] = lpips_loss
            loss = loss + self.lpips_weight * lpips_loss
        if self.auxiliary_weight > 0:
            auxiliary_loss = F.mse_loss(inverted_roi, lensed_roi)
            output["auxiliary_loss"] = auxiliary_loss
            loss = loss + self.auxiliary_weight * auxiliary_loss
        output["loss"] = loss
        return output

    @staticmethod
    def to_lpips_range(image):
        return image.clamp(0, 1) * 2 - 1
