import warnings

import comet_ml
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from to_my.project_dl_homework_final_clean.src.datasets.data_utils import get_dataloaders
from to_my.project_dl_homework_final_clean.src.trainer import ReconstructionTrainer
from to_my.project_dl_homework_final_clean.src.utils.init_utils import set_random_seed, setup_experiment_dir, setup_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="leadmm5_prepost")
def main(config):
    set_random_seed(config.trainer.seed)
    save_dir, _ = setup_experiment_dir(config)
    logger = setup_logging(save_dir)
    project_config = OmegaConf.to_container(config, resolve=True)
    writer = instantiate(
        config.writer,
        logger=logger,
        project_config=project_config,
        _recursive_=False,
    )
    if config.trainer.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.trainer.device)
    logger.info(f"используем device: {device}")
    dataloaders = get_dataloaders(config)
    model = instantiate(config.model).to(device)
    criterion = instantiate(config.loss).to(device)
    metrics = instantiate(config.metrics)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = instantiate(config.optimizer, params=parameters)
    trainer = ReconstructionTrainer(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        save_dir=save_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
