"""
app.py — API REST para clasificación de géneros de películas
MIAD Uniandes — Proyecto 2

Framework : FastAPI
Modelo    : TF-IDF (50 000 features, bigramas) + Logistic Regression One-vs-Rest
Géneros   : 24 clases (multilabel)
Métrica   : ROC AUC macro ≈ 0.899

Endpoints:
  GET  /               → información de la API
  GET  /health         → estado del servicio
  POST /predict        → predicción para una película
  POST /predict_batch  → predicción para varias películas
"""

import joblib
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Carga del modelo al iniciar la aplicación
# ──────────────────────────────────────────────
MODEL_FILE = "model_pipeline.pkl"
artifact: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el artefacto una sola vez al arrancar el servidor."""
    global artifact
    if not os.path.exists(MODEL_FILE):
        raise RuntimeError(
            f"Artefacto '{MODEL_FILE}' no encontrado. "
            "Ejecuta primero: python train.py"
        )
    artifact = joblib.load(MODEL_FILE)
    print(f"✅ Modelo cargado — {len(artifact['genres'])} géneros disponibles.")
    yield
    artifact.clear()


# ──────────────────────────────────────────────
# Aplicación FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="Movie Genre Classification API",
    description=(
        "API REST para predecir la probabilidad de que una película pertenezca "
        "a cada uno de los 24 géneros cinematográficos, dada su sinopsis (plot) "
        "y título. Desarrollada como parte del Proyecto 2 de MIAD Uniandes.\n\n"
        "**Modelo:** TF-IDF + Regresión Logística (One-vs-Rest multilabel)\n"
        "**Métrica:** ROC AUC macro ≈ 0.899 en validación"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Schemas Pydantic
# ──────────────────────────────────────────────
class MovieInput(BaseModel):
    title: str = Field(..., example="The Dark Knight", description="Título de la película")
    plot: str = Field(
        ...,
        example=(
            "When the menace known as the Joker wreaks havoc and chaos on the people "
            "of Gotham, Batman must accept one of the greatest psychological and "
            "physical tests of his ability to fight injustice."
        ),
        description="Sinopsis o plot de la película",
    )
    year: Optional[int] = Field(None, example=2008, description="Año de estreno (opcional, no usado en el modelo)")


class GenreProbabilities(BaseModel):
    title: str
    probabilities: Dict[str, float]
    top_genres: List[str]


class BatchInput(BaseModel):
    movies: List[MovieInput] = Field(..., min_length=1, max_length=100)


class BatchOutput(BaseModel):
    predictions: List[GenreProbabilities]
    total_movies: int
    processing_time_seconds: float


# ──────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────
def _build_text(movie: MovieInput) -> str:
    """Combina título y sinopsis igual que en el entrenamiento."""
    return f"{movie.title} {movie.plot}"


def _predict_single(text: str) -> Dict[str, float]:
    """Devuelve un dict {género: probabilidad} para un texto."""
    pipeline = artifact["pipeline"]
    genres = artifact["genres"]
    probs = pipeline.predict_proba([text])[0]
    return {f"p_{g}": float(round(p, 6)) for g, p in zip(genres, probs)}


def _top_genres(probs: Dict[str, float], n: int = 3) -> List[str]:
    """Retorna los N géneros con mayor probabilidad."""
    sorted_genres = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return [g.replace("p_", "") for g, _ in sorted_genres[:n]]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """Información general de la API."""
    return {
        "nombre": "Movie Genre Classification API",
        "version": "1.0.0",
        "descripcion": (
            "Predice la probabilidad de pertenencia a cada uno de los 24 "
            "géneros cinematográficos dada la sinopsis y el título de una película."
        ),
        "generos_disponibles": artifact.get("genres", []),
        "endpoints": {
            "GET  /":               "Información de la API",
            "GET  /health":         "Estado del servicio",
            "POST /predict":        "Predicción para una película",
            "POST /predict_batch":  "Predicción para varias películas (máx. 100)",
            "GET  /docs":           "Documentación interactiva Swagger",
        },
        "modelo": "TF-IDF (bigramas, 50k features) + Logistic Regression One-vs-Rest",
        "metrica_referencia": "ROC AUC macro ≈ 0.899",
        "proyecto": "MIAD Uniandes — Proyecto 2",
    }


@app.get("/health", tags=["Info"])
def health():
    """Verifica que el servicio y el modelo estén disponibles."""
    if not artifact:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    return {
        "status": "ok",
        "model_loaded": True,
        "num_genres": len(artifact.get("genres", [])),
    }


@app.post("/predict", response_model=GenreProbabilities, tags=["Predicción"])
def predict(movie: MovieInput):
    """
    Predice las probabilidades de género para **una** película.

    - **title**: Título de la película (requerido)
    - **plot**: Sinopsis o descripción de la trama (requerido)
    - **year**: Año de estreno (opcional, no afecta la predicción)

    Retorna las probabilidades para los 24 géneros y los 3 géneros más probables.
    """
    if not artifact:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    try:
        text = _build_text(movie)
        probs = _predict_single(text)
        top = _top_genres(probs)
        return GenreProbabilities(title=movie.title, probabilities=probs, top_genres=top)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {exc}")


@app.post("/predict_batch", response_model=BatchOutput, tags=["Predicción"])
def predict_batch(batch: BatchInput):
    """
    Predice las probabilidades de género para **múltiples** películas (máx. 100).

    Recibe una lista de películas y retorna las probabilidades de género para cada una,
    junto con los 3 géneros más probables por película.
    """
    if not artifact:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    try:
        t0 = time.time()
        pipeline = artifact["pipeline"]
        genres = artifact["genres"]

        texts = [_build_text(m) for m in batch.movies]
        all_probs = pipeline.predict_proba(texts)

        results = []
        for movie, probs_arr in zip(batch.movies, all_probs):
            probs_dict = {f"p_{g}": float(round(p, 6)) for g, p in zip(genres, probs_arr)}
            top = _top_genres(probs_dict)
            results.append(
                GenreProbabilities(title=movie.title, probabilities=probs_dict, top_genres=top)
            )

        elapsed = round(time.time() - t0, 4)
        return BatchOutput(
            predictions=results,
            total_movies=len(results),
            processing_time_seconds=elapsed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en la predicción batch: {exc}")
