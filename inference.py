import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from PIL import Image
from tqdm.auto import tqdm

from to_my.project_dl_homework_final_clean.src.datasets.data_utils import get_dataloaders
from to_my.project_dl_homework_final_clean.src.utils.init_utils import set_random_seed, setup_quiet_external_logging

warnings.filterwarnings("ignore", category=UserWarning)


def tensor_to_image(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)
    array = (tensor.numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    setup_quiet_external_logging()
    set_random_seed(config.inferencer.seed)
    if config.inferencer.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.inferencer.device)
    dataloaders = get_dataloaders(config)
    model = instantiate(config.model).to(device)
    checkpoint = torch.load(
        config.inferencer.checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    output_dir = Path(config.inferencer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for _, loader in dataloaders.items():
            for batch in tqdm(loader, desc="инференс"):
                for key in config.inferencer.device_tensors:
                    batch[key] = batch[key].to(device)
                outputs = model(
                    lensless=batch["lensless"],
                    psf=batch["psf"],
                )
                reconstructions = outputs["reconstruction_roi"]
                for image_id, reconstruction in zip(batch["ids"], reconstructions):
                    tensor_to_image(reconstruction).save(output_dir / f"{image_id}.png")


if __name__ == "__main__":
    main()
