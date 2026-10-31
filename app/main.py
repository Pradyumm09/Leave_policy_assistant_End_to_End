from dotenv import load_dotenv

# Load environment variables BEFORE importing application modules
load_dotenv()

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import CONFIG
from app.core.logger import logger


app = FastAPI(
    title=CONFIG["app"]["name"],
    version=CONFIG["app"]["version"],
    description=CONFIG["app"]["description"],
)


app.include_router(router)


@app.on_event("startup")
async def startup_event():
    logger.info("Leave Policy Assistant API starting...")


@app.get("/")
async def root():
    return {
        "message": "Leave Policy Assistant API",
        "version": CONFIG["app"]["version"],
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": CONFIG["app"]["name"],
    }
