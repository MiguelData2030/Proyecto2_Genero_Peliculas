"""
train.py — Entrenamiento del modelo de clasificacion de generos de peliculas
MIAD Uniandes — Proyecto 2

Pipeline: TF-IDF (20 000 features, bigramas) + Regresion Logistica One-vs-Rest
Metrica de referencia: ROC AUC macro aprox 0.899 en validacion

Este script se ejecuta durante el despliegue en Render
para generar el artefacto model_pipeline.pkl.
"""

import ast
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
SEED = 42
DATA_URL = (
    "https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip"
)
OUTPUT_FILE = "model_pipeline.pkl"

# Hiperparametros optimos (RandomizedSearchCV 50 iter, 5 folds)
BEST_PARAMS = {
    "max_features": 20_000,   # reducido de 50k para menor uso de memoria
    "ngram_range":  (1, 2),
    "min_df":       2,
    "max_df":       0.95,
    "C":            1.0,
}


# ──────────────────────────────────────────────
# Funciones
# ──────────────────────────────────────────────
def load_data(url: str) -> pd.DataFrame:
    """Descarga y prepara el conjunto de entrenamiento."""
    print(f"[1/4] Descargando datos desde GitHub ...")
    df = pd.read_csv(url, encoding="UTF-8", index_col=0)

    # Parsear genres de string a lista
    df["genres"] = df["genres"].apply(ast.literal_eval)

    # Texto combinado: titulo + sinopsis
    df["title"] = df["title"].astype(str)
    df["plot"]  = df["plot"].astype(str)
    df["info_pelicula"] = df["title"] + " " + df["plot"]

    # Eliminar duplicados exactos
    df["genres_tuple"] = df["genres"].apply(tuple)
    cols_dup = [c for c in df.columns if c != "genres"]
    df = df.drop_duplicates(subset=cols_dup, keep="first")
    df = df.drop(columns=["genres_tuple"])

    print(f"    Registros cargados: {df.shape[0]}")
    return df


def train(df: pd.DataFrame):
    """Entrena el pipeline y lo guarda en disco."""
    # Binarizacion multilabel
    print("[2/4] Binarizando etiquetas multilabel ...")
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["genres"])
    print(f"    Generos ({len(mlb.classes_)}): {list(mlb.classes_)}")

    # Pipeline TF-IDF + Regresion Logistica
    print("[3/4] Entrenando pipeline TF-IDF + Logistic Regression ...")
    pipeline = Pipeline(
        [
            (
                "vectorizador",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    sublinear_tf=True,
                    max_features=BEST_PARAMS["max_features"],
                    ngram_range=BEST_PARAMS["ngram_range"],
                    min_df=BEST_PARAMS["min_df"],
                    max_df=BEST_PARAMS["max_df"],
                ),
            ),
            (
                "clasificador",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1000,
                        C=BEST_PARAMS["C"],
                        random_state=SEED,
                    ),
                    n_jobs=1,   # 1 job en free tier (0.1 CPU)
                ),
            ),
        ]
    )
    pipeline.fit(df["info_pelicula"], y)
    print("    Entrenamiento completado.")

    # Serializar artefacto
    print(f"[4/4] Guardando artefacto en '{OUTPUT_FILE}' ...")
    artifact = {
        "pipeline": pipeline,
        "mlb":      mlb,
        "genres":   list(mlb.classes_),
    }
    joblib.dump(artifact, OUTPUT_FILE)
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"    Artefacto guardado: {size_mb:.1f} MB")
    return artifact


# ──────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ENTRENAMIENTO MODELO — CLASIFICACION GENERO PELICULAS")
    print("=" * 60)
    try:
        df = load_data(DATA_URL)
        train(df)
        print("=" * 60)
        print("LISTO — modelo guardado correctamente.")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR durante el entrenamiento: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
