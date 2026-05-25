"""
app.py — API REST para clasificacion de generos de peliculas
MIAD Uniandes — Proyecto 2

Framework : FastAPI
Modelo    : TF-IDF + Logistic Regression One-vs-Rest (multilabel)
Generos   : 24 clases
Metrica   : ROC AUC macro aprox 0.899

Endpoints:
  GET  /               -> informacion de la API
  GET  /health         -> estado del servicio
  POST /predict        -> prediccion para una pelicula
  POST /predict_batch  -> prediccion para varias peliculas
"""

import joblib
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Carga del modelo al iniciar la aplicacion
# ──────────────────────────────────────────────
MODEL_FILE = "model_pipeline.pkl"
artifact: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global artifact
    if not os.path.exists(MODEL_FILE):
        print(f"ADVERTENCIA: '{MODEL_FILE}' no encontrado. Ejecuta python train.py primero.")
    else:
        artifact = joblib.load(MODEL_FILE)
        print(f"Modelo cargado — {len(artifact['genres'])} generos disponibles.")
    yield
    artifact.clear()


# ──────────────────────────────────────────────
# Aplicacion FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="Movie Genre Classification API",
    description=(
        "API REST para predecir la probabilidad de que una pelicula pertenezca "
        "a cada uno de los 24 generos cinematograficos, dada su sinopsis y titulo. "
        "Desarrollada como parte del Proyecto 2 de MIAD Uniandes.\n\n"
        "**Modelo:** TF-IDF + Regresion Logistica (One-vs-Rest multilabel)\n"
        "**Metrica:** ROC AUC macro aprox 0.899 en validacion"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Schemas Pydantic
# ──────────────────────────────────────────────
class MovieInput(BaseModel):
    title: str = Field(..., example="The Dark Knight")
    plot: str  = Field(
        ...,
        example=(
            "When the menace known as the Joker wreaks havoc and chaos on the people "
            "of Gotham, Batman must accept one of the greatest psychological and "
            "physical tests of his ability to fight injustice."
        ),
    )
    year: Optional[int] = Field(None, example=2008)


class GenreProbabilities(BaseModel):
    title: str
    probabilities: Dict[str, float]
    top_genres: List[str]


class BatchInput(BaseModel):
    movies: List[MovieInput]


class BatchOutput(BaseModel):
    predictions: List[GenreProbabilities]
    total_movies: int
    processing_time_seconds: float


# ──────────────────────────────────────────────
# Utilidades internas
# ──────────────────────────────────────────────
def _check_model():
    if not artifact:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. El archivo model_pipeline.pkl no fue encontrado."
        )


def _build_text(movie: MovieInput) -> str:
    return f"{movie.title} {movie.plot}"


def _predict_single(text: str) -> Dict[str, float]:
    pipeline = artifact["pipeline"]
    genres   = artifact["genres"]
    probs    = pipeline.predict_proba([text])[0]
    return {f"p_{g}": float(round(float(p), 6)) for g, p in zip(genres, probs)}


def _top_genres(probs: Dict[str, float], n: int = 3) -> List[str]:
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return [k.replace("p_", "") for k, _ in sorted_items[:n]]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """Informacion general de la API."""
    return {
        "nombre":  "Movie Genre Classification API",
        "version": "1.0.0",
        "descripcion": (
            "Predice la probabilidad de pertenencia a 24 generos cinematograficos "
            "dada la sinopsis y el titulo de una pelicula."
        ),
        "generos_disponibles": artifact.get("genres", []),
        "endpoints": {
            "GET  /":              "Informacion de la API",
            "GET  /health":        "Estado del servicio",
            "POST /predict":       "Prediccion para una pelicula",
            "POST /predict_batch": "Prediccion para varias peliculas (max 100)",
            "GET  /docs":          "Documentacion interactiva Swagger",
        },
        "modelo":            "TF-IDF (bigramas, 20k features) + Logistic Regression One-vs-Rest",
        "metrica_referencia": "ROC AUC macro aprox 0.899",
        "proyecto":          "MIAD Uniandes - Proyecto 2",
    }


@app.get("/health", tags=["Info"])
def health():
    """Verifica que el servicio y el modelo esten disponibles."""
    return {
        "status":       "ok" if artifact else "model_not_loaded",
        "model_loaded": bool(artifact),
        "num_genres":   len(artifact.get("genres", [])),
    }


@app.post("/predict", response_model=GenreProbabilities, tags=["Prediccion"])
def predict(movie: MovieInput):
    """
    Predice las probabilidades de genero para **una** pelicula.

    - **title**: Titulo de la pelicula (requerido)
    - **plot**: Sinopsis de la trama (requerido)
    - **year**: Año de estreno (opcional)
    """
    _check_model()
    try:
        text  = _build_text(movie)
        probs = _predict_single(text)
        top   = _top_genres(probs)
        return GenreProbabilities(title=movie.title, probabilities=probs, top_genres=top)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en la prediccion: {exc}")


@app.post("/predict_batch", response_model=BatchOutput, tags=["Prediccion"])
def predict_batch(batch: BatchInput):
    """
    Predice las probabilidades de genero para **multiples** peliculas (max 100).
    """
    _check_model()
    if len(batch.movies) > 100:
        raise HTTPException(status_code=400, detail="Maximo 100 peliculas por request.")
    try:
        t0       = time.time()
        pipeline = artifact["pipeline"]
        genres   = artifact["genres"]

        texts     = [_build_text(m) for m in batch.movies]
        all_probs = pipeline.predict_proba(texts)

        results = []
        for movie, probs_arr in zip(batch.movies, all_probs):
            probs_dict = {
                f"p_{g}": float(round(float(p), 6))
                for g, p in zip(genres, probs_arr)
            }
            results.append(
                GenreProbabilities(
                    title=movie.title,
                    probabilities=probs_dict,
                    top_genres=_top_genres(probs_dict),
                )
            )

        return BatchOutput(
            predictions=results,
            total_movies=len(results),
            processing_time_seconds=round(time.time() - t0, 4),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en prediccion batch: {exc}")
