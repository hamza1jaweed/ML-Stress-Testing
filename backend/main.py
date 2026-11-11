"""FastAPI application entrypoint.

Run locally:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Market Regime Detection & Portfolio Stress-Testing Platform",
    description=(
        "Unsupervised regime classification (GMM + Markov transitions) and "
        "Monte Carlo / historical-replay portfolio stress testing."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": "regime-stress-platform",
        "docs": "/docs",
        "health": "/api/health",
    }
