from collections import defaultdict
from itertools import islice

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from to_my.project_dl_homework_final_clean.src.datasets.data_utils import inf_loop


class RunningAverage:
    def __init__(self):
        self.values = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, name, value, n=1):
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        self.values[name] += float(value) * n
        self.counts[name] += n

    def result(self):
        return {
            name: self.values[name] / max(self.counts[name], 1)
            for name in self.values
        }


class ReconstructionTrainer:
    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        dataloaders,
        logger,
        writer,
        save_dir,
    ):
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.config = config
        self.cfg = config.trainer
        self.device = device
        self.train_loader = inf_loop(dataloaders["train"])
        self.eval_loaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.logger = logger
        self.writer = writer
        self.save_dir = save_dir
        self.global_step = 0
        self.best_psnr = float("-inf")

    def train(self):
        for epoch in range(1, self.cfg.n_epochs + 1):
            self.train_epoch(epoch)
            if self.global_step >= self.cfg.total_steps:
                break
        self.save_checkpoint("model_last.pth")
        self.writer.close()

    def train_epoch(self, epoch):
        self.model.train()
        avg = RunningAverage()
        progress = tqdm(
            range(self.cfg.epoch_len),
            desc=f"обучение epoch {epoch}",
            dynamic_ncols=True,
            mininterval=2.0,
        )
        for _ in progress:
            batch = self.move_batch_to_device(next(self.train_loader))
            losses, outputs = self.train_step(batch)
            self.global_step += 1
            for name, value in losses.items():
                avg.update(name, value)
            if self.global_step % self.cfg.log_step == 0:
                scalars = avg.result()
                self.writer.set_step(self.global_step, "train")
                self.writer.add_scalars(scalars)
                progress.set_postfix(loss=f"{scalars['loss']:.4f}")
                avg = RunningAverage()
            if self.global_step % self.cfg.image_log_step == 0:
                self.log_images(batch, outputs, "train")
            if self.global_step % self.cfg.save_step == 0:
                self.save_checkpoint(f"checkpoint-step{self.global_step}.pth")
            if self.global_step % self.cfg.eval_step == 0:
                self.evaluate_all(epoch)
            if self.global_step >= self.cfg.total_steps:
                break

    def train_step(self, batch):
        outputs = self.model(
            lensless=batch["lensless"],
            psf=batch["psf"],
        )
        losses = self.criterion(
            lensed_roi=batch["lensed_roi"],
            reconstruction_roi=outputs["reconstruction_roi"],
            inverted_roi=outputs["inverted_roi"],
        )
        self.optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        return losses, outputs

    @torch.no_grad()
    def evaluate_all(self, epoch):
        for part, loader in self.eval_loaders.items():
            logs, last_batch, last_outputs = self.evaluate(part, loader)
            self.writer.set_step(self.global_step, part)
            self.writer.add_scalars(logs)
            self.log_images(last_batch, last_outputs, part)
            message = ", ".join(f"{key}: {value:.4f}" for key, value in logs.items())
            self.logger.info(f"epoch {epoch} step {self.global_step} {part}: {message}")
            if logs["PSNR"] > self.best_psnr:
                self.best_psnr = logs["PSNR"]
                self.save_checkpoint("model_best.pth")

    @torch.no_grad()
    def evaluate(self, part, loader):
        self.model.eval()
        avg = RunningAverage()
        last_batch = {}
        last_outputs = {}
        total_batches = min(len(loader), int(self.cfg.max_eval_batches))
        self.logger.info(f"оценка {part}: {total_batches} batches")
        progress = tqdm(
            enumerate(islice(loader, total_batches)),
            desc=f"оценка {part}",
            dynamic_ncols=True,
            mininterval=2.0,
            leave=True,
            total=total_batches,
        )
        for _, batch in progress:
            batch = self.move_batch_to_device(batch)
            outputs = self.model(
                lensless=batch["lensless"],
                psf=batch["psf"],
            )
            losses = self.criterion(
                lensed_roi=batch["lensed_roi"],
                reconstruction_roi=outputs["reconstruction_roi"],
                inverted_roi=outputs["inverted_roi"],
            )
            for name, value in losses.items():
                avg.update(name, value)
            for metric in self.metrics["inference"]:
                avg.update(
                    metric.name,
                    metric(
                        lensed_roi=batch["lensed_roi"],
                        reconstruction_roi=outputs["reconstruction_roi"],
                    ),
                )
            last_batch = batch
            last_outputs = outputs
        self.model.train()
        return avg.result(), last_batch, last_outputs

    def move_batch_to_device(self, batch):
        for key in self.cfg.device_tensors:
            batch[key] = batch[key].to(self.device)
        return batch

    def log_images(self, batch, outputs, mode):
        self.writer.set_step(self.global_step, mode)
        self.writer.add_image("lensless", batch["lensless"][0])
        self.writer.add_image("target_roi", batch["lensed_roi"][0])
        self.writer.add_image("inverted_roi", outputs["inverted_roi"][0])
        self.writer.add_image("reconstruction_roi", outputs["reconstruction_roi"][0])

    def checkpoint_state(self):
        state = {
            "step": self.global_step,
            "model": self.model.state_dict(),
            "config": self.config,
            "best_psnr": self.best_psnr,
            "optimizer": self.optimizer.state_dict(),
        }
        return state

    def save_checkpoint(self, name):
        path = self.save_dir / name
        torch.save(self.checkpoint_state(), path)
        self.logger.info(f"checkpoint сохранен: {path}")
