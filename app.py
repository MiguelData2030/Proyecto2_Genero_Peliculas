"""
app.py - API REST para clasificacion de generos de peliculas
MIAD Uniandes - Proyecto 2

Framework : FastAPI
Modelo    : TF-IDF + Logistic Regression One-vs-Rest (multilabel)
Generos   : 24 clases
Metrica   : ROC AUC macro aprox 0.899
"""

import joblib
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Carga del modelo al iniciar
# ──────────────────────────────────────────────
MODEL_FILE = "model_pipeline.pkl"
artifact: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global artifact
    if not os.path.exists(MODEL_FILE):
        print(f"ADVERTENCIA: '{MODEL_FILE}' no encontrado.")
    else:
        artifact = joblib.load(MODEL_FILE)
        print(f"Modelo cargado — {len(artifact['genres'])} generos disponibles.")
    yield
    artifact.clear()


# ──────────────────────────────────────────────
# Metadata OpenAPI (estilo Proyecto Andes)
# ──────────────────────────────────────────────
tags_metadata = [
    {
        "name": "Información",
        "description": "Endpoints de información general y estado del servicio.",
    },
    {
        "name": "Predicción",
        "description": (
            "Endpoints de predicción de géneros cinematográficos. "
            "Reciben el título y la sinopsis de una película y retornan "
            "las probabilidades para cada uno de los **24 géneros**."
        ),
    },
]

app = FastAPI(
    title="API Clasificación Géneros - Películas",
    description=(
        "Predice la probabilidad de que una película pertenezca a cada uno de los "
        "**24 géneros cinematográficos**, dada su sinopsis y título. "
        "Desarrollada como parte del **Proyecto 2 de MIAD Uniandes**.\n\n"
        "**Modelo:** TF-IDF (bigramas, 20 000 features) + "
        "Regresión Logística One-vs-Rest (multilabel)\n\n"
        "**Métrica:** ROC AUC macro ≈ 0.899 en validación\n\n"
        "**Equipo:** Adolfo Ramírez Moreno · Gisell Zarina Gutiérrez Fernandez · "
        "Miguel Ángel Londoño Díaz · Winston Andrés Licona Briceño"
    ),
    version="2.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
    contact={
        "name": "MIAD Uniandes - Proyecto 2",
        "url": "https://github.com/MiguelData2030/Proyecto2_Genero_Peliculas",
    },
)


# ──────────────────────────────────────────────
# Schemas Pydantic
# ──────────────────────────────────────────────
class MovieInput(BaseModel):
    title: str = Field(
        ...,
        example="The Dark Knight",
        description="Título de la película.",
    )
    plot: str = Field(
        ...,
        example=(
            "When the menace known as the Joker wreaks havoc and chaos on the people "
            "of Gotham, Batman must accept one of the greatest psychological and "
            "physical tests of his ability to fight injustice."
        ),
        description="Sinopsis o descripción de la trama de la película.",
    )
    year: Optional[int] = Field(
        None,
        example=2008,
        description="Año de estreno (opcional, no afecta la predicción).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "The Dark Knight",
                "plot": (
                    "When the menace known as the Joker wreaks havoc and chaos "
                    "on the people of Gotham, Batman must accept one of the greatest "
                    "psychological and physical tests of his ability to fight injustice."
                ),
                "year": 2008,
            }
        }
    }


class GenreProbabilities(BaseModel):
    title: str = Field(..., description="Título de la película.")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilidades para cada uno de los 24 géneros (p_Action, p_Drama, ...).",
    )
    top_genres: List[str] = Field(
        ...,
        description="Los 3 géneros con mayor probabilidad predicha.",
    )


class BatchInput(BaseModel):
    movies: List[MovieInput] = Field(
        ...,
        description="Lista de películas a clasificar (máximo 100).",
        example=[
            {
                "title": "Toy Story",
                "plot": (
                    "A cowboy doll is profoundly threatened and jealous when a new "
                    "spaceman figure supplants him as top toy in a boy's room."
                ),
                "year": 1995,
            },
            {
                "title": "Psycho",
                "plot": (
                    "A secretary embezzles money from her employer's client, goes on "
                    "the run, and checks into a remote motel run by a young man under "
                    "the domination of his mother."
                ),
                "year": 1960,
            },
        ],
    )


class BatchOutput(BaseModel):
    predictions: List[GenreProbabilities]
    total_movies: int = Field(..., description="Total de películas procesadas.")
    processing_time_seconds: float = Field(..., description="Tiempo de procesamiento en segundos.")


# ──────────────────────────────────────────────
# Utilidades internas
# ──────────────────────────────────────────────
def _check_model():
    if not artifact:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. El archivo model_pipeline.pkl no fue encontrado.",
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
@app.get(
    "/",
    tags=["Información"],
    summary="Raíz",
    response_description="Información general de la API.",
)
def root():
    """
    Retorna información general sobre la API: nombre, versión, modelo,
    métrica de referencia, géneros disponibles y endpoints.
    """
    return {
        "nombre":  "API Clasificación Géneros - Películas",
        "version": "2.0.0",
        "descripcion": (
            "Predice la probabilidad de pertenencia a 24 géneros cinematográficos "
            "dada la sinopsis y el título de una película."
        ),
        "modelo": "TF-IDF (bigramas, 20k features) + Logistic Regression One-vs-Rest",
        "metrica_referencia": "ROC AUC macro ≈ 0.899",
        "proyecto": "MIAD Uniandes - Proyecto 2",
        "generos_disponibles": artifact.get("genres", []),
        "endpoints": {
            "GET  /":              "Información de la API",
            "GET  /health":        "Estado del servicio",
            "POST /predict":       "Predicción para una película",
            "POST /predict_batch": "Predicción para varias películas (máx. 100)",
            "GET  /docs":          "Documentación interactiva Swagger UI",
            "GET  /redoc":         "Documentación ReDoc",
        },
    }


@app.get(
    "/health",
    tags=["Información"],
    summary="Salud",
    response_description="Estado actual del servicio y del modelo cargado.",
)
def health():
    """
    Verifica que el servicio esté operativo y que el modelo de clasificación
    haya sido cargado correctamente en memoria.
    """
    return {
        "status":       "ok" if artifact else "model_not_loaded",
        "model_loaded": bool(artifact),
        "num_genres":   len(artifact.get("genres", [])),
    }


@app.post(
    "/predict",
    response_model=GenreProbabilities,
    tags=["Predicción"],
    summary="Predecir géneros de una película",
    response_description="Probabilidades de género y top 3 géneros predichos.",
)
def predict(movie: MovieInput):
    """
    Recibe el título y la sinopsis de **una** película y devuelve la probabilidad
    de pertenencia a cada uno de los 24 géneros cinematográficos.

    **Entradas requeridas:**
    - **title**: Título de la película.
    - **plot**: Sinopsis o descripción de la trama.

    **Salida:**
    - Diccionario con probabilidades para los 24 géneros (`p_Action`, `p_Drama`, ...).
    - Lista con los 3 géneros de mayor probabilidad (`top_genres`).

    **Modelo:** TF-IDF + Logistic Regression One-vs-Rest | ROC AUC macro ≈ 0.899
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
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {exc}")


@app.post(
    "/predict_batch",
    response_model=BatchOutput,
    tags=["Predicción"],
    summary="Predecir géneros de múltiples películas",
    response_description="Predicciones para cada película junto con el tiempo de procesamiento.",
)
def predict_batch(batch: BatchInput):
    """
    Recibe una lista de películas (máximo **100**) y devuelve las probabilidades
    de género para cada una, junto con los 3 géneros más probables.

    **Entrada:**
    - Lista de objetos película, cada uno con `title` y `plot`.

    **Salida:**
    - Lista de predicciones con probabilidades y top géneros por película.
    - Tiempo total de procesamiento en segundos.

    **Modelo:** TF-IDF + Logistic Regression One-vs-Rest | ROC AUC macro ≈ 0.899
    """
    _check_model()
    if len(batch.movies) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 películas por request.")
    try:
        t0       = time.time()
        pipeline = artifact["pipeline"]
        genres   = artifact["genres"]
        texts    = [_build_text(m) for m in batch.movies]
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
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {exc}")
