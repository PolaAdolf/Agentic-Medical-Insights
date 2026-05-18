"""
main.py
=======
FastAPI Entry Point # Agentic Medical Insights
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    print("Agentic-Medical-Insights API starting up…")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Agentic Medical Insights",
    description=(
        "An agentic AI pipeline that OCRs medical lab reports, "
        "extracts structured context, searches PubMed, and generates "
        "personalized evidence-based clinical reports using Gemma 4 + DSPy."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "service": "Agentic Medical Insights"}
