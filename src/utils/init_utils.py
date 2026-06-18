import logging
import os
import random
import shutil
import uuid

import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.utils.io_utils import ROOT_PATH


def setup_quiet_external_logging():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for logger_name in [
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "datasets",
        "urllib3",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_id(length=16):
    return uuid.uuid4().hex[:length]


def setup_logging(save_dir):
    setup_quiet_external_logging()
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lensless")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(save_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def setup_experiment_dir(config):
    load_dotenv(ROOT_PATH / ".env")
    run_id = generate_id(32)
    save_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.set_struct(config, False)
    config.writer.run_id = run_id
    OmegaConf.set_struct(config, True)
    OmegaConf.save(config, save_dir / "config.yaml")
    return save_dir, run_id
