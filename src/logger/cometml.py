import os
from datetime import datetime

import comet_ml
from dotenv import load_dotenv

from to_my.project_dl_homework_final_clean.src.utils.io_utils import ROOT_PATH


class CometMLWriter:
    def __init__(
        self,
        logger,
        project_config,
        project_name,
        workspace=None,
        run_id=None,
        run_name=None,
        mode="online",
        log_code=False,
        log_graph=False,
    ):
        load_dotenv(ROOT_PATH / ".env")
        self.logger = logger
        self.step = 0
        self.mode = "train"
        self.timer = datetime.now()
        api_key = os.getenv("COMET_API_KEY")
        if mode == "offline":
            self.exp = comet_ml.OfflineExperiment(
                project_name=project_name,
                workspace=workspace,
                experiment_key=run_id,
                log_code=log_code,
            )
        else:
            self.exp = comet_ml.Experiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace,
                experiment_key=run_id,
                log_code=log_code,
                log_graph=log_graph,
                auto_metric_logging=False,
                auto_param_logging=False,
            )
        if run_name:
            self.exp.set_name(run_name)
        self.exp.log_parameters(project_config)

    def set_step(self, step, mode="train"):
        previous_step = self.step
        self.step = step
        self.mode = mode
        if step > previous_step:
            duration = datetime.now() - self.timer
            seconds = duration.total_seconds()
            if seconds > 0:
                self.add_scalar("steps_per_sec", (step - previous_step) / seconds)
            self.timer = datetime.now()

    def _name(self, name):
        return f"{self.mode}/{name}"

    def add_scalar(self, name, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        self.exp.log_metric(self._name(name), value, step=self.step)

    def add_scalars(self, scalars):
        for name, value in scalars.items():
            self.add_scalar(name, value)

    def add_image(self, name, image):
        if hasattr(image, "detach"):
            image = image.detach().cpu().float()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        array = image.clamp(0, 1).numpy()
        self.exp.log_image(array, name=f"{self._name(name)}.png", step=self.step)

    def close(self):
        self.exp.end()
