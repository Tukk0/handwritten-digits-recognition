# Handwritten Digit Recognition with Deep Learning

**Листов Тихон Андреевич, МФТИ, MLOps, весна 2026**

## 1. Постановка задачи

Система классификации рукописных цифр (0–9) на растровых изображениях. На вход подаётся чёрно-белое изображение размером 28×28 пикселей, содержащее одну цифру, написанную от руки. На выходе — предсказанный класс (цифра 0–9) с уверенностью модели.

Проект включает полный MLOps-пайплайн: загрузка и управление данными через DVC, обучение через PyTorch Lightning, логирование через MLFlow/TensorBoard, экспорт модели в ONNX, развёртывание inference-сервера через NVIDIA Triton.

## 2. Входные и выходные данные

**Вход:** изображение в форматах PNG/JPG, grayscale, размер 28×28 пикселей. Возможна подача фото произвольного размера — предобработка масштабирует и нормализует его автоматически.

**Выход:**

- Предсказанный класс (int: 0–9).
- softmax-вероятности для каждого из 10 классов.

## 3. Метрики

Основная метрика — **accuracy**. Дополнительно:

- **Per-class recall и precision** — для анализа баланса между классами.
- **Macro-F1 score** — гармоническое среднее precision и recall по всем классам.
- **Confusion matrix** — визуализация ошибок.

Планируемый target — **accuracy ≥ 98.5%** на test set.

## 4. Разделение данных

Стратифицированное разделение 60/20/20 (train/validation/test).

## 5. Датасет

**Основной датасет:** MNIST.

- **Источник:** `https://huggingface.co/datasets/ylecun/mnist`
- **Объём:** 60 000 изображений для тренировки, 10 000 для тестирования — ~70 МБ (gzipped).
- **Формат:** grayscale PNG/IDX, 28×28 пикселей, интенсивность 0–255.
- **Дата публикации:** 1998 г.
- **Особенности:** цифры написаны посетителями Американского бюро переписи населения и студентами.

## 6. Моделирование

### Базовый бейзлайн

**Логистическая регрессия** с softmax. ~92% accuracy.

### Основная модель

**LeNet-5 модифицированная** (BatchNorm, dropout, AdamW):

```
Conv2d(1, 32, 3) → BatchNorm → ReLU → MaxPool
Conv2d(32, 64, 3) → BatchNorm → ReLU → MaxPool
Flatten → Linear(3136, 128) → ReLU → Dropout(0.5) → Linear(128, 10)
```

### Тренировочный пайплайн (PyTorch Lightning):

- **Предобработка:** torchvision.transforms — нормализация (mean=0.1307, std=0.3081), аргументация.
- **Обучение:** PyTorch Lightning — модуль, DataModule, Trainer с EarlyStopping, MLFlow/WandB.
- **Оптимизатор:** AdamW с weight decay=1e-2, LR scheduling (CosineAnnealingLR).
- **Loss:** CrossEntropyLoss.

## 7. Внедрение

### Production preparation

- **ONNX** — универсальный инференсный формат.
- **TensorRT** — оптимизированное runtime-исполнение на NVIDIA GPU.

### Inference server

**NVIDIA Triton Inference Server** — ONNX модель загружается в Triton.

### Telegram-бот

Telegram-бот получает изображение, передаёт его в Triton inference-сервер и возвращает предсказание.

## 8. Setup

### Через Docker (рекомендуется)

Самый быстрый способ запуска — через **Docker**:

```bash
# Построить образ
docker build -t digit-recognition .

# Тренировка (5 эпох):
docker run --rm digit-recognition train 5

# Тесты:
docker run --rm digit-recognition test

# Лайнинг:
docker run --rm digit-recognition lint

# Полный пайплайн (train -> onnx -> convert):
docker run --rm digit-recognition train-onnx-inference
```

### Через **run.sh** (локальная установка)

Установите зависимости:

```bash
pip install poetry && poetry install
pre-commit install
```

Запуск:

```bash
./run.sh train        # 5 эпох
./run.sh train 30     # 30 эпох
./run.sh onnx         # export to ONNX
./run.sh test         # pytest
./run.sh lint         # ruff check
./run.sh docker       # docker build
./run.sh docker-all   # train -> onnx через Docker
```

Через **Hydra** (для кастомных конфигов):

```bash
source .venv/bin/activate
python train_main.py
python train_main.py training.max_epochs=50 training.optimizer.lr=0.0005
```

### Локальная установка

```bash
pip install poetry
poetry install
pre-commit install
pre-commit run -a
```

## 9. Тренировка

### Через Docker (рекомендуется)

```bash
docker run --rm digit-recognition train 10     # 10 эпох
docker run --rm digit-recognition test          # тесты
docker run --rm digit-recognition lint          # лайнинг
docker run --rm digit-recognition train-onnx-inference  # полный пайплайн
```

### Через run.sh

```bash
./run.sh train        # 5 эпох (по умолчанию)
./run.sh train 30     # 30 эпох
./run.sh test         # pytest
./run.sh lint         # ruff check
```

### Локально (через Hydra)

```bash
python train_main.py
python train_main.py training.max_epochs=50 training.optimizer.lr=0.0005
```

### Логи

- **MLFlow** на `http://127.0.0.1:8080` (метрики, параметры, git commit).
- **TensorBoard** в `logs/run_<timestamp>/`.
- **Checkpoints модели** сохраняются в `checkpoints/`.

## 10. Инференс

### Через Docker

```bash
docker run --rm digit-recognition onnx                    # train -> onnx
docker run --rm digit-recognition inference checkpoints/best.pt test.png  # инференс
```

### Через run.sh

```bash
./run.sh onnx     # export last checkpoint to ONNX + verify
./run.sh inference    # show inference ready (then use infer.py directly)
```

### Локально

```bash
python infer.py predict checkpoints/best.pt test.png
python infer.py onnx --model checkpoints/best.pt --onnx model.onnx
python scripts/convert.py export_onnx checkpoints/best.pt model.onnx
python scripts/convert.py test_onnx_consistency checkpoints/best.pt model.onnx
tritonserver --model-repo models/digit_recognition --http-port 8000 --grpc-port 8001
```

## 11. Конфигурация

Все гиперпараметры лежат в `configs/` — иерархические YAML-файлы Hydra:

```yaml
├── configs/
│   ├── data.yaml          # batch_size, workers, augmentation params
│   ├── model.yaml         # архитектура, слои, dropout, batch_norm
│   ├── training.yaml      # optimizer, scheduler, early_stopping, devices
│   ├── logging.yaml       # mlflow_uri, wandb_project, mlflow_experiment_name
│   ├── inference.yaml     # model_path, image_size, device, onnx/trt paths
│   ├── train.yaml         # defaults group для тренировки
│   └── infer.yaml         # defaults group для инференса
```

Переопределение через CLI:

```bash
python scripts/train.py training.max_epochs=50 training.optimizer.lr=0.0005
```

## 12. Production preparation

### Через Docker

```bash
docker run --rm digit-recognition onnx    # train -> onnx -> verify
```

### Локально

```bash
python scripts/convert.py export_onnx checkpoints/best.pt model.onnx
python scripts/convert.py test_onnx_consistency checkpoints/best.pt model.onnx

# TensorRT (требуется NVIDIA GPU + tensorrt installed)
./deploy_tensorrt.sh model.onnx model.trt 1
```

## 13. Telegram-бот

### Через Docker

```bash
docker compose up --build
```

Настроено автоматически: бот запустится только после `triton` healthcheck.

### Локально

```bash
cp .env.local .env.local.bak
# В .env.local.bak: export TELEGRAM_BOT_TOKEN=<your_token>
python scripts/main.py
```

> **Примечание**: Telegram-бот не имеет CLI-аргументов. Указать endpoint Triton-сервера можно через переменную окружения:
>
> ```bash
> TRITON_ENDPOINT="http://localhost:8000/v2/models/digit_recognition/infer" python scripts/main.py
> ```

## 14. Docker

### Быстрый старт

```bash
docker build -t digit-recognition .

# Полный пайплайн (train + onnx + convert)
docker run --rm digit-recognition train-onnx-inference

# Тренировка (10 эпох)
docker run --rm digit-recognition train 10

# Тесты
docker run --rm digit-recognition test

# Лайнинг
docker run --rm digit-recognition lint

# Только Triton + Telegram bot
docker compose up --build -d
```

### Dockerfile

```dockerfile
FROM python:3.11-slim
# ... все зависимости для проекта
```

### Команды Docker

| Команда                                             | Результат               |
| --------------------------------------------------- | ----------------------- |
| `docker run digit-recognition train 5`              | Training 5 epochs       |
| `docker run digit-recognition test`                 | pytest                  |
| `docker run digit-recognition lint`                 | ruff check              |
| `docker run digit-recognition onnx`                 | export to ONNX + verify |
| `docker run digit-recognition train-onnx-inference` | Full pipeline           |

### Docker Compose (production)

```bash
docker compose up --build --detach
```

Запускает:

1. **Triton Inference Server** — с healthcheck (port 8000/8001)
2. **Telegram Bot** — запускается только после healthy Triton

## 15. Структура проекта

```
digit-recognition/
├── configs/
│   ├── data.yaml          # параметры загрузки и структуры dataloader
│   ├── model.yaml         # архитектура модели
│   ├── training.yaml      # гиперпараметры обучения (optimizer, lr, batch_size)
│   ├── logging.yaml       # MLFlow/WandB конфигурация
│   ├── inference.yaml     # параметры инференса и конвертации
│   ├── train.yaml         # defaults group для тренировки
│   └── infer.yaml         # defaults group для инференса
├── digit_recognition/
│   ├── __init__.py        # package marker + __all__
│   ├── commands.py        # единая точка входа CLI (train, onnx, test, lint)
│   ├── data_module.py     # LightningDataModule для загрузки данных
│   ├── model.py           # PyTorch Lightning модуль, архитектура CNN
│   ├── preprocessing.py   # transforms и нормализация
│   ├── inference.py       # инференс (load, predict, ONNX)
│   └── download_data.py   # download MNIST via DVC Python API
├── scripts/
│   ├── convert.py         # конвертация модели в ONNX
│   └── main.py            # Telegram-бот (входная точка)
├── models/
│   └── digit_recognition/ # Triton Model Repository
│       ├── 1/
│       │   └── model.onnx
│       └── config.pbtxt
├── train_main.py           # точка входа в тренировку (Hydra + Lightning)
├── infer.py               # точка входа в инференс (public API, CLI)
├── deploy_tensorrt.sh     # shell-скрипт для TensorRT конвертации
├── pyproject.toml         # зависимости (Poetry)
├── .gitignore
├── .pre-commit-config.yaml
├── .env.local             # переменные окружения (Telegram Token)
├── docker-compose.yaml    # оркестрация Triton + Telegram-бот
├── Dockerfile             # контейнеризация Telegram-бота
├── dvc.yaml               # DVC pipeline
├── .dvc/
│   ├── config             # хранилища данных и моделей
│   └── ...
├── data/                  # данные под управлением DVC (не в git)
├── checkpoints/           # модели .ckpt
├── plots/                 # графики обучения
├── logs/                  # TensorBoard лог
└── mlruns/                # MLFlow лог
```
