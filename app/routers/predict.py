from fastapi import APIRouter, UploadFile, File, Form

from .. import config
from .. import schemas
from ..ml.model import load_model, preprocess_image, prepare_additional_data, predict
from ..storage.image_storage import process_image_file

import torch
from datetime import datetime
from io import BytesIO


router = APIRouter(prefix="/predict", tags=["predict"])

# Загрузка модели при старте приложения
cv_model_path = config.DEFAULT_CV_WEIGHTS
mlp_model_path = config.DEFAULT_MLP_WEIGHTS
classifier_model_path = config.DEFAULT_CLASSIFIER_WEIGHTS
model = load_model(cv_model_path, mlp_model_path, classifier_model_path)

@router.post("/")
async def make_prediction(
    image: UploadFile = File(...),
    gender: str = Form(...),
    birthdate: str = Form(...),  # ISO format string
    localization: str = Form(...),
):
    try:
        # Валидация даты рождения
        try:
            birth_date_obj = datetime.fromisoformat(birthdate)
            if birth_date_obj > datetime.now():
                return {
                    "ok": False,
                    "status": 400,
                    "error": "Birth date cannot be in the future"
                }
        except ValueError:
            return {
                "ok": False,
                "status": 400,
                "error": "Invalid date format. Use ISO format (YYYY-MM-DD)"
            }

        # Обработка изображения
        image_data = await image.read()
        processed_image = process_image_file(UploadFile(file=BytesIO(image_data), filename=image.filename))
        
        # Предсказание с использованием модели
        image_tensor = preprocess_image(processed_image)
        additional_tensor = prepare_additional_data(gender, birthdate, localization)
        predicted_classes_probabilities = predict(model, image_tensor, additional_tensor)

        prediction = torch.argmax(predicted_classes_probabilities, dim=0).item()
        prediction_probability = predicted_classes_probabilities[prediction].item()
        
        return {
            "ok": True,
            "status": 200,
            "data": {
                "prediction": prediction,
                "probability": prediction_probability 
            }
        }
    
    except Exception as e:
        return {
            "ok": False,
            "status": 500,
            "error": str(e)
        }
