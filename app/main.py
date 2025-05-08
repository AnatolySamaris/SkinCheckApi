from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from .database import engine
# from . import models
from .routers import users, results, predict

# Создание таблиц в базе данных
# models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Pytorch Model API",
    description="API for serving pytorch model and managing user data",
    version="0.1.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
# app.include_router(users.router)
# app.include_router(results.router)
app.include_router(predict.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Pytorch Model API"}