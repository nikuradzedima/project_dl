## Что реализовано

- `ADMM-100`: фиксированный ADMM на 100
- `LeADMM-20`: обучаемый ADMM на 20 итераций
- `LeADMM-5 + pre/post`: ADDM примерно на 8M параметров с DRUNet preprocessor и DRUNet postprocessor
- `LeADMM-5 + pre`: ADDM вариант только с preprocessor.
- `LeADMM-5 + post`: ADDM вариант только с postprocessor.


## Структура

```text
train.py
evaluate.py
inference.py
calculate_metrics.py
requirements.txt
src/
  configs/        Hydra-конфиги
  datasets/       DigiCam и CustomDirDataset
  logger/         Comet ML
  loss/           MSE + LPIPS
  metrics/        PSNR, MSE, SSIM, LPIPS
  model/          ADMM, DRUNet
  trainer/       
  utils/          
scripts/
  download_checkpoint.py
notebooks/
  train_lensless.ipynb
  demo_lensless.ipynb
reports/
  REPORT.md
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Нужно создать `.env` в корне проекта с ключом для CometML

```text
COMET_API_KEY=...
```

## Обучение

Основной запуск обучения лучшей модели

```bash
python3 train.py -cn=leadmm5_prepost \
  trainer.device=cuda \
  dataloader.batch_size=2 \
  dataloader.eval_batch_size=1 \
  dataloader.num_workers=2 \
  trainer.max_eval_batches=100 \
  writer.mode=online \
  writer.run_name=leadmm5_prepost_exp \
  trainer.override=true
```

Запуск обучения остальных моделей

```bash
python3 train.py -cn=leadmm20 \
  trainer.device=cuda \
  dataloader.batch_size=1 \
  dataloader.eval_batch_size=1 \
  dataloader.num_workers=2 \
  writer.mode=online \
  writer.run_name=leadmm20_exp \
  trainer.override=true

python3 train.py -cn=leadmm5_pre \
  trainer.device=cuda \
  writer.mode=online \
  writer.run_name=leadmm5_pre_exp \
  trainer.override=true

python3 train.py -cn=leadmm5_post \
  trainer.device=cuda \
  writer.mode=online \
  writer.run_name=leadmm5_post_exp \
  trainer.override=true

python3 evaluate.py -cn=admm100 \
  trainer.device=cuda \
  dataloader.batch_size=1 \
  dataloader.eval_batch_size=1 \
  dataloader.num_workers=2 \
  trainer.max_eval_batches=100
```

Финальная оценка лучшей модели

```bash
python3 evaluate.py -cn=leadmm5_prepost \
  evaluation.checkpoint_path=saved/leadmm5_prepost_exp/model_best.pth \
  trainer.device=cuda \
  dataloader.eval_batch_size=1 \
  dataloader.num_workers=2 \
  trainer.max_eval_batches=1500
```

## Инференс

Датасет должен иметь структуру

```text
custom_dataset/
  lensless/
    ImageID1.png
  masks/
    ImageID1.npy
  lensed/
    ImageID1.png
```

Запуск инференса

```bash
python3 inference.py \
  inferencer.checkpoint_path=saved/leadmm5_prepost_exp/model_best.pth \
  datasets.test.root=/path/to/custom_dataset \
  inferencer.output_dir=outputs/reconstructions \
  inferencer.device=cuda
```

## Метрики 

```bash
python3 calculate_metrics.py \
  data_dir=/path/to/custom_dataset \
  prediction_dir=outputs/reconstructions \
  device=cuda
```

Будут посчитаны следующие метрики `MSE`, `PSNR`, `SSIM`, `LPIPS`

