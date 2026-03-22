from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from backend.pipeline import verify_article
from backend.ocr import extract_text_from_image
from backend.image_verify import verify_image_with_clip
from backend.task import enqueue_article, get_job_result
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class NewsRequest(BaseModel):
    text: str
    url: str | None = None
@app.post("/verify")
def verify(news: NewsRequest):
    result = verify_article(news.text, news.url)
    return {
        **result,
        "meta": {
            "engine": "semantic-similarity + dataset",
            "mode": "sync"
        }
    }
@app.post("/verify-async")
def verify_async(news: NewsRequest):
    job_id = enqueue_article(news.text, news.url)
    return {"job_id": job_id}
@app.get("/result/{job_id}")
def get_result(job_id: str):
    result = get_job_result(job_id)
    if result["status"] == "finished":
        result["result"]["meta"] = {
            "engine": "rq-worker pipeline",
            "mode": "async"
        }
    return result
@app.post("/verify-image")
def verify_image(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    extracted_text = extract_text_from_image(file_path)
    analysis = verify_article(extracted_text)
    image_result = verify_image_with_clip(file_path, extracted_text)
    os.remove(file_path)
    return {
        "analysis": analysis,
        "image_verification": image_result,
        "ocr_text": extracted_text,
        "meta": {
            "pipeline": ["OCR", "Semantic Matching", "CLIP"],
            "note": "Demo-level verification, not forensic proof"
        }
    }