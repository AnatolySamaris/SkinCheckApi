from ultralytics import YOLO

import torch
from torch import nn
from torchvision import transforms

from datetime import datetime
from dateutil.relativedelta import relativedelta
from PIL import Image as PILImage
from io import BytesIO


class MLP(nn.Module):
    def __init__(self, input_size=15, num_classes=7, hidden_size=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


class CancerModel(nn.Module):
    def __init__(self, yolo, mlp, num_classes=7):
        super().__init__()
        self.yolo = yolo.model
        self.mlp = mlp
        self.classifier = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, tabular):
        with torch.no_grad():
            yolo_out = self.yolo(image)
            assert isinstance(yolo_out, tuple), "YOLO model is not in eval mode."
            yolo_logits = yolo_out[1][0]
        mlp_out = self.mlp(tabular)
        combined = torch.cat([mlp_out, yolo_logits], dim=0)
        result = self.classifier(combined)
        return result


def load_model(cv_model_path: str, mlp_model_path: str, classifier_model_path: str):
    yolo = YOLO(cv_model_path, task="classify")
    mlp = MLP()
    mlp.load_state_dict(torch.load(mlp_model_path, map_location=torch.device('cpu')))
    model = CancerModel(
        yolo=yolo,
        mlp=mlp,
    )
    model.load_state_dict(torch.load(classifier_model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_image(image_data: bytes) -> torch.Tensor:
    image = PILImage.open(BytesIO(image_data))
    preprocess = transforms.Compose([
        transforms.Resize(640),
        transforms.ToTensor(),
    ])
    return preprocess(image).unsqueeze(0)


def prepare_additional_data(gender: str, birthdate: str, localization: str) -> torch.Tensor:
    # Кодируем пол
    assert gender in ['Мужской', 'Женский'], 'Gender must be only "Мужской" or "Женский".'
    encoded_gender = int(gender == 'Мужской')

    # Получаем возраст
    birthdate_datetime = datetime.fromisoformat(birthdate)
    current_date = datetime.now()
    diff = relativedelta(current_date, birthdate_datetime)
    age = diff.years / 100 # Нормализованный возраст

    # Кодируем локализацию в формате OneHotEncoding
    localization_list = [
        "Ладонь, подошва или ноготь", "Спина", "Грудь",
        "Ухо", "Лицо", "Стопа", "Гениталии", "Кисть руки", 
        "Бедро, колено или голень", "Шея", "Скальп (волосистая часть головы)",
        "Живот, бока или пах", "Плечо, локоть или предплечье"
    ]
    encoded_localization = []
    for loc in localization_list:
        encoded_localization.append(int(loc == localization))
    
    # Объединение в единый тензор
    encoded_additional = torch.Tensor([age, encoded_gender] + encoded_localization)

    return encoded_additional


def predict(model, image_tensor: torch.Tensor, additional_tensor: torch.Tensor) -> list:
    assert additional_tensor.size(0) == 15, f"Additional tensor size is {additional_tensor.size(0)}, must be 15."
    image = image_tensor.float()
    additional = additional_tensor.float()
    with torch.no_grad():
        outputs = model(image, additional)
        probabilities = torch.softmax(outputs, dim=0)
        percent_classes = probabilities * 100
    return percent_classes
