from torch import nn

from to_my.project_dl_homework_final_clean.src.model.drunet import DRUNet
from to_my.project_dl_homework_final_clean.src.model.fft_admm import UnrolledADMM


class LenslessReconstructionModel(nn.Module):
    def __init__(
        self,
        num_iterations,
        learnable_admm=True,
        use_preprocessor=False,
        use_postprocessor=False,
        pre_channels=(32, 64, 128, 256),
        post_channels=(32, 64, 128, 256),
        init_mu=1e-4,
        init_tau=2e-4,
        roi_top=80,
        roi_left=100,
        roi_height=200,
        roi_width=266,
    ):
        super().__init__()
        self.roi_top = roi_top
        self.roi_left = roi_left
        self.roi_height = roi_height
        self.roi_width = roi_width
        if use_preprocessor:
            self.preprocessor = DRUNet(channels=pre_channels, residual=True)
        else:
            self.preprocessor = nn.Identity()
        self.inverter = UnrolledADMM(
            num_iterations=num_iterations,
            learnable=learnable_admm,
            init_mu=init_mu,
            init_tau=init_tau,
        )
        if use_postprocessor:
            self.postprocessor = DRUNet(channels=post_channels, residual=True)
        else:
            self.postprocessor = nn.Identity()

    def forward(
        self,
        lensless,
        psf,
    ):
        preprocessed = self.preprocessor(lensless).clamp(0, 1)
        inverted = self.inverter(preprocessed, psf).clamp(0, 1)
        reconstruction = self.postprocessor(inverted).clamp(0, 1)
        return {
            "inverted_roi": self.crop_roi(inverted),
            "reconstruction_roi": self.crop_roi(reconstruction),
        }

    def crop_roi(self, image):
        top = self.roi_top
        left = self.roi_left
        return image[
            ...,
            top : top + self.roi_height,
            left : left + self.roi_width,
        ]
