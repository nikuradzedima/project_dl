import numpy as np
import torch
from to_my.project_dl_homework_final_clean.src.datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from to_my.project_dl_homework_final_clean.lensless_helpers.preprocessor import get_dataset_object, get_roi


class DigiCamDataset(Dataset):
    def __init__(
        self,
        split,
        repo_id="bezzam/DigiCam-Mirflickr-MultiMask-10K",
        cache_dir=None,
        mask_cache_dir=None,
        limit=None,
    ):
        self.split = split
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.mask_cache_dir = mask_cache_dir
        self.dataset = load_dataset(repo_id, split=split, cache_dir=cache_dir)
        if limit is not None:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))
        self.psf_cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        mask_label = int(sample["mask_label"])
        mask = self.load_mask(mask_label)
        lensed, lensless, psf = get_dataset_object(
            sample["lensed"], sample["lensless"], mask
        )
        lensed_roi = get_roi(lensed)
        return {
            "id": f"{self.split}_{index:06d}",
            "lensless": self.to_chw(lensless),
            "lensed": self.to_chw(lensed),
            "lensed_roi": self.to_chw(lensed_roi),
            "psf": self.to_chw(psf.squeeze(0)),
        }

    def load_mask(self, label):
        if label not in self.psf_cache:
            filename = f"masks/mask_{label}.npy"
            path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=filename,
                cache_dir=self.mask_cache_dir or self.cache_dir,
            )
            self.psf_cache[label] = np.load(path)
        return self.psf_cache[label]

    @staticmethod
    def to_chw(image):
        tensor = torch.as_tensor(image).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        return tensor.permute(2, 0, 1).contiguous()
