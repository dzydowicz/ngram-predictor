#!/usr/bin/env python
"""
FastAPI wrapper exposing POST /predict
Expected payload: {"context": ["up", "to", "3", "words"]}
Return shape     : {"prediction": "nextword"}
"""
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr
import uvicorn

from src.service import NGramService, DEFAULT_ORDERS

CORPUS = os.getenv("CORPUS_PATH", "data/data.txt")
ALPHA = float(os.getenv("ALPHA", 1.0))

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.svc = NGramService(CORPUS, DEFAULT_ORDERS, alpha=ALPHA)
    print("[api] Models ready.")
    yield

    print("[api] Shutdown complete.")

app = FastAPI(title="N-gram predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    context: List[constr(strip_whitespace=True, min_length=1)]

class PredictionResponse(BaseModel):
    prediction: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(ctx: PredictionRequest, request: Request):
    svc: NGramService = request.app.state.svc
    if svc is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        pred = svc.predict_next(ctx.context)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"prediction": pred or ""}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
