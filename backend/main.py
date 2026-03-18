from fastapi import FastAPI
from backend.pipeline import verify_article
app = FastAPI()
@app.post("/verify")
async def verify(data: dict):
    text = data["text"]
    result = verify_article(text)
    return result