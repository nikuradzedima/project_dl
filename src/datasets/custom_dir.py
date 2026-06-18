from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from to_my.project_dl_homework_final_clean.lensless_helpers.preprocessor import get_dataset_object, get_roi


class CustomDirDataset(Dataset):
    def __init__(self, root, limit=None):
        self.root = Path(root)
        self.lensless_dir = self.root / "lensless"
        self.masks_dir = self.root / "masks"
        self.lensed_dir = self.root / "lensed"
        self.ids = sorted(path.stem for path in self.lensless_dir.glob("*.png"))
        if limit is not None:
            self.ids = self.ids[:limit]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        lensless = Image.open(self.lensless_dir / f"{image_id}.png").convert("RGB")
        mask = np.load(self.masks_dir / f"{image_id}.npy")
        if self.lensed_dir.exists() and (self.lensed_dir / f"{image_id}.png").exists():
            lensed = Image.open(self.lensed_dir / f"{image_id}.png").convert("RGB")
            lensed, lensless, psf = get_dataset_object(lensed, lensless, mask)
            item = {
                "lensed": self.to_chw(lensed),
                "lensed_roi": self.to_chw(get_roi(lensed)),
            }
        else:
            lensless_np = np.asarray(lensless)
            fallback_lensed = Image.fromarray(lensless_np)
            _, lensless, psf = get_dataset_object(fallback_lensed, lensless, mask)
            item = {}
        item.update(
            {
                "id": image_id,
                "lensless": self.to_chw(lensless),
                "psf": self.to_chw(psf.squeeze(0)),
            }
        )
        return item

    @staticmethod
    def to_chw(image):
        tensor = torch.as_tensor(image).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        return tensor.permute(2, 0, 1).contiguous()
