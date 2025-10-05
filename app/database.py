from fastapi import FastAPI
from .database import Base, engine
from .auth import router as auth_router

app = FastAPI(title="AQI App with Auth")

Base.metadata.create_all(bind=engine)  # ← создаёт таблицы автоматически

app.include_router(auth_router)
