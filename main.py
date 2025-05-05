import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from utils.inference import KatutuboLLM
from utils.similarity_search import UPVectorDB

# Load environment variables
load_dotenv(override=True)
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("GPU_ID", "")

# Initialize services
model = KatutuboLLM()
vector_db = UPVectorDB()
is_ready = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("request_logger")

# Lifespan for FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    global is_ready
    is_ready = True
    yield

# Initialize FastAPI
app = FastAPI(
    title="Katutubo LLM API",
    description="API for Katutubo LLM Inference.",
    version="1.0.0",
    lifespan=lifespan,
)

# Request model
class Prompt(BaseModel):
    prompt: str
    history: list

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Katutubo LLM Inference API!"}

@app.get("/healthz")
async def healthz():
    return {"ready": is_ready}

@app.post("/infer")
async def model_inference(Request: Prompt):
    try:
        logger.info(f"Prompt: {Request.prompt}")
        similar_faq = vector_db.similarity_search(Request.prompt)
        logger.info(f"Similar FAQ: {similar_faq}")

        response = model.infer(Request.prompt, Request.history, similar_faq) if similar_faq else model.infer(Request.prompt, Request.history)
        return {"response": response}

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
