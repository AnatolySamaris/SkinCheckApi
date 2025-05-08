from pathlib import Path
from typing import Optional
import os

# Базовый путь проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Пути к весам обученной модели
DEFAULT_CV_WEIGHTS = BASE_DIR / "app" / "ml" / "weights" / "yolo_model.pt"
DEFAULT_MLP_WEIGHTS = BASE_DIR / "app" / "ml" / "weights" / "mlp_model.pt"
DEFAULT_CLASSIFIER_WEIGHTS = BASE_DIR / "app" / "ml" / "weights" / "classifier_model.pt"
