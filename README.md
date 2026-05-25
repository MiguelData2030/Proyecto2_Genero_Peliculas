# 🎬 Movie Genre Classification API

> **MIAD Uniandes — Proyecto 2**  
> API REST para predecir la probabilidad de que una película pertenezca a cada uno de los 24 géneros cinematográficos, dada su sinopsis y título.

---

## 📌 Descripción

El modelo utiliza un pipeline de **TF-IDF + Regresión Logística One-vs-Rest (multilabel)** entrenado sobre ~7 895 películas del dataset del Profesor Fabio González (Uniandes). El texto de entrada combina el título y la sinopsis de la película.

- **Métrica de referencia:** ROC AUC macro ≈ 0.899 en validación
- **Géneros predichos (24):** Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, History, Horror, Music, Musical, Mystery, News, Romance, Sci-Fi, Short, Sport, Thriller, War, Western

---

## 🚀 Despliegue en Render

La API está disponible en:

```
https://movie-genre-classification-api.onrender.com
```

### Flujo de despliegue automático

```
git push → Render detecta cambios
         → pip install -r requirements.txt
         → python train.py          # entrena y guarda model_pipeline.pkl
         → uvicorn app:app ...      # levanta la API
```

---

## 🔌 Endpoints

| Método | Ruta            | Descripción                                  |
|--------|-----------------|----------------------------------------------|
| GET    | `/`             | Información general de la API                |
| GET    | `/health`       | Estado del servicio                          |
| POST   | `/predict`      | Predicción para una película                 |
| POST   | `/predict_batch`| Predicción para múltiples películas (máx.100)|
| GET    | `/docs`         | Documentación interactiva Swagger            |

---

## 📥 Ejemplos de uso

### POST `/predict`

**Request:**
```json
{
  "title": "The Dark Knight",
  "plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
  "year": 2008
}
```

**Response:**
```json
{
  "title": "The Dark Knight",
  "probabilities": {
    "p_Action":    0.521,
    "p_Crime":     0.487,
    "p_Thriller":  0.463,
    "p_Drama":     0.341,
    "p_Adventure": 0.198,
    "...": "..."
  },
  "top_genres": ["Action", "Crime", "Thriller"]
}
```

---

### POST `/predict_batch`

**Request:**
```json
{
  "movies": [
    {
      "title": "Toy Story",
      "plot": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room."
    },
    {
      "title": "Psycho",
      "plot": "A secretary embezzles money from her employer's client, goes on the run, and checks into a remote motel run by a young man under the domination of his mother."
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "title": "Toy Story",
      "probabilities": { "p_Animation": 0.72, "p_Comedy": 0.61 },
      "top_genres": ["Animation", "Comedy", "Family"]
    },
    {
      "title": "Psycho",
      "probabilities": { "p_Horror": 0.68, "p_Thriller": 0.65 },
      "top_genres": ["Horror", "Thriller", "Drama"]
    }
  ],
  "total_movies": 2,
  "processing_time_seconds": 0.032
}
```

---

## 🛠️ Ejecución local

```bash
# 1. Clonar el repositorio
git clone https://github.com/MiguelData2030/Proyecto2_Genero_Peliculas.git
cd Proyecto2_Genero_Peliculas

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrenar el modelo (genera model_pipeline.pkl)
python train.py

# 4. Levantar la API
uvicorn app:app --reload

# 5. Abrir documentación interactiva
# http://127.0.0.1:8000/docs
```

---

## 🗂️ Estructura del proyecto

```
Proyecto2_Genero_Peliculas/
├── app.py                  # API FastAPI (endpoints y lógica de inferencia)
├── train.py                # Entrenamiento del modelo y serialización
├── requirements.txt        # Dependencias Python
├── render.yaml             # Configuración despliegue Render.com
├── README.md               # Documentación
└── .gitignore
```

---

## 🔧 Arquitectura del modelo

```
Entrada (título + sinopsis)
       ↓
TF-IDF Vectorizer
  • max_features = 50 000
  • ngram_range  = (1, 2)
  • min_df = 1  |  max_df = 0.95
  • sublinear_tf = True
  • stop_words = "english"
       ↓
OneVsRestClassifier
  └── LogisticRegression (liblinear, C=1.0)
       ↓
Probabilidades para 24 géneros
```

---

## 👥 Autores

Proyecto 2 — MIAD Uniandes  
Datos: Fabio González, Ph.D. & John Arevalo — https://arxiv.org/abs/1702.01992
