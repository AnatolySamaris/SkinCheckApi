from PIL import Image as PILImage
from io import BytesIO
from fastapi import UploadFile
import base64

def process_image_file(file: UploadFile):
    # Чтение файла
    contents = file.file.read()
    
    try:
        image = PILImage.open(BytesIO(contents))
        image.thumbnail((640, 640)) # Изменение размера изображения
        
        # Конвертация обратно в bytes
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        processed_image = buffered.getvalue()
    except Exception as e:
        raise ValueError(f"Ошибка обработки изображения: {str(e)}")
    
    return processed_image

def image_to_base64(image_data: bytes) -> str:
    return base64.b64encode(image_data).decode('utf-8')

def base64_to_image(base64_str: str) -> bytes:
    return base64.b64decode(base64_str)