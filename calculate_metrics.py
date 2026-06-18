from pathlib import Path

import hydra
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from src.datasets import CustomDirDataset
from src.metrics.image import LPIPSMetric, MSEMetric, PSNRMetric, SSIMMetric
from src.utils.init_utils import setup_quiet_external_logging


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    return pil_to_tensor(image).float() / 255.0


@hydra.main(version_base=None, config_path="src/configs", config_name="metrics")
def main(config):
    setup_quiet_external_logging()
    pred_dir = Path(config.prediction_dir)
    metrics = [
        MSEMetric(),
        PSNRMetric(),
        SSIMMetric(),
        LPIPSMetric(device=config.device),
    ]
    values = {metric.name: [] for metric in metrics}
    dataset = CustomDirDataset(config.data_dir)
    items = [(item["id"], item["lensed_roi"]) for item in dataset]
    for image_id, target in items:
        pred_path = pred_dir / f"{image_id}.png"
        prediction = load_rgb(pred_path)
        lensed_roi = target.unsqueeze(0).to(config.device)
        reconstruction_roi = prediction.unsqueeze(0).to(config.device)
        for metric in metrics:
            values[metric.name].append(
                metric(
                    lensed_roi=lensed_roi,
                    reconstruction_roi=reconstruction_roi,
                )
            )
    for name, scores in values.items():
        value = sum(scores) / max(len(scores), 1)
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
