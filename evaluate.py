import warnings

from itertools import islice

import hydra
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_random_seed, setup_quiet_external_logging

warnings.filterwarnings("ignore", category=UserWarning)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)


@hydra.main(version_base=None, config_path="src/configs", config_name="admm100")
def main(config):
    setup_quiet_external_logging()
    set_random_seed(config.trainer.seed)
    if config.trainer.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.trainer.device)
    dataset = instantiate(config.datasets.test)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataloader.eval_batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        collate_fn=collate_fn,
    )
    model = instantiate(config.model).to(device)
    if config.evaluation.checkpoint_path is not None:
        load_checkpoint(model, config.evaluation.checkpoint_path, device)
    model.eval()
    criterion = instantiate(config.loss).to(device)
    metrics = instantiate(config.metrics)["inference"]
    totals = {"loss": 0.0}
    counts = {"loss": 0}
    for metric in metrics:
        totals[metric.name] = 0.0
        counts[metric.name] = 0
    total_batches = min(len(dataloader), int(config.trainer.max_eval_batches))
    with torch.no_grad():
        for batch in tqdm(
            islice(dataloader, total_batches),
            desc="оценка",
            dynamic_ncols=True,
            mininterval=2.0,
            leave=True,
            total=total_batches,
        ):
            for key in config.trainer.device_tensors:
                batch[key] = batch[key].to(device)
            outputs = model(
                lensless=batch["lensless"],
                psf=batch["psf"],
            )
            losses = criterion(
                lensed_roi=batch["lensed_roi"],
                reconstruction_roi=outputs["reconstruction_roi"],
                inverted_roi=outputs["inverted_roi"],
            )
            totals["loss"] += float(losses["loss"].detach().cpu().item())
            counts["loss"] += 1
            for metric in metrics:
                value = metric(
                    lensed_roi=batch["lensed_roi"],
                    reconstruction_roi=outputs["reconstruction_roi"],
                )
                totals[metric.name] += value
                counts[metric.name] += 1
    for name, total in totals.items():
        value = total / max(counts[name], 1)
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
