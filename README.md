# 🎬 API Clasificación Géneros - Películas

> **MIAD Uniandes — Proyecto 2**  
> API REST para predecir la probabilidad de que una película pertenezca a cada uno de los 24 géneros cinematográficos, dada su sinopsis y título.

🔗 **API en producción:** [https://proyecto2-genero-peliculas.onrender.com](https://proyecto2-genero-peliculas.onrender.com)  
📖 **Documentación Swagger:** [https://proyecto2-genero-peliculas.onrender.com/docs](https://proyecto2-genero-peliculas.onrender.com/docs)

---

## 🖥️ Guía paso a paso: cómo usar la API

> La API ya está desplegada y funcionando. No necesitas instalar nada. Solo sigue estos pasos desde tu navegador.

---

### Paso 1 — Abre la documentación interactiva

Haz clic en este enlace o pégalo en tu navegador:

👉 **[https://proyecto2-genero-peliculas.onrender.com/docs](https://proyecto2-genero-peliculas.onrender.com/docs)**

Verás una pantalla azul y verde llamada **Swagger UI**. Esta es la interfaz oficial de la API — desde aquí puedes probarla sin escribir código.

> ⚠️ **Importante:** La API está alojada en el plan gratuito de Render. Si lleva varios minutos sin usarse, entra en "reposo". La **primera vez que abras la página puede tardar 30–60 segundos** en cargar. Es normal — espera hasta que aparezca la interfaz completa.

---

### Paso 2 — Verifica que el modelo está cargado

Antes de hacer predicciones, confirma que la API está activa:

1. En la pantalla de Swagger, busca la sección **"Información"**
2. Haz clic en la fila verde que dice **`GET /health`**
3. Haz clic en el botón **"Try it out"** (aparece a la derecha)
4. Haz clic en el botón azul **"Execute"**
5. Baja hasta **"Response body"** y verifica que ves:

```json
{
  "status": "ok",
  "model_loaded": true,
  "num_genres": 24
}
```

✅ Si `"model_loaded"` es `true` y `"num_genres"` es `24`, la API está lista para predecir.

---

### Paso 3 — Predice el género de una película

1. En la pantalla de Swagger, busca la sección **"Predicción"**
2. Haz clic en la fila azul que dice **`POST /predict`**
3. Haz clic en el botón **"Try it out"**
4. Verás un cuadro de texto con un JSON de ejemplo. **Reemplaza** ese contenido con los datos de tu película. Por ejemplo:

```json
{
  "title": "The Dark Knight",
  "plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
  "year": 2008
}
```

> 💡 `title` = título de la película  
> 💡 `plot` = sinopsis o descripción de la trama (en inglés da mejores resultados)  
> 💡 `year` = año (opcional, no afecta la predicción)

5. Haz clic en el botón azul **"Execute"**
6. Baja hasta **"Response body"** y verás el resultado:

```json
{
  "title": "The Dark Knight",
  "probabilities": {
    "p_Action": 0.3437,
    "p_Thriller": 0.3235,
    "p_Drama": 0.3214,
    "p_Adventure": 0.2286,
    ...
  },
  "top_genres": ["Action", "Thriller", "Drama"]
}
```

**¿Qué significa cada campo?**
- **`probabilities`** → probabilidad (entre 0 y 1) de que la película pertenezca a cada uno de los 24 géneros
- **`top_genres`** → los 3 géneros con mayor probabilidad (los más probables)

---

### Paso 4 — Predice géneros para varias películas a la vez

Si quieres clasificar varias películas en una sola llamada:

1. En la sección **"Predicción"**, haz clic en la fila azul **`POST /predict_batch`**
2. Haz clic en **"Try it out"**
3. En el cuadro de texto, ingresa una lista de películas (máximo 100):

```json
{
  "movies": [
    {
      "title": "Toy Story",
      "plot": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.",
      "year": 1995
    },
    {
      "title": "The Shining",
      "plot": "A family heads to an isolated hotel for the winter where a sinister presence influences the father into violence, while his psychic son sees horrific forebodings from both past and future.",
      "year": 1980
    }
  ]
}
```

4. Haz clic en **"Execute"**
5. El resultado incluirá las predicciones para cada película y el tiempo total de procesamiento:

```json
{
  "predictions": [
    {
      "title": "Toy Story",
      "probabilities": { "p_Animation": 0.72, "p_Comedy": 0.61, ... },
      "top_genres": ["Animation", "Comedy", "Family"]
    },
    {
      "title": "The Shining",
      "probabilities": { "p_Horror": 0.74, "p_Thriller": 0.58, ... },
      "top_genres": ["Horror", "Thriller", "Drama"]
    }
  ],
  "total_movies": 2,
  "processing_time_seconds": 0.031
}
```

---

### Resumen rápido

| Qué quieres hacer | Endpoint | Cómo |
|---|---|---|
| Verificar que la API funciona | `GET /health` | Abre `/docs` → sección Información → `/health` → Try it out → Execute |
| Clasificar **una** película | `POST /predict` | Abre `/docs` → sección Predicción → `/predict` → Try it out → pega el JSON → Execute |
| Clasificar **varias** películas | `POST /predict_batch` | Abre `/docs` → sección Predicción → `/predict_batch` → Try it out → pega el JSON → Execute |

---

## 📌 Descripción

El modelo utiliza un pipeline de **TF-IDF + Regresión Logística One-vs-Rest (multilabel)** entrenado sobre ~7 895 películas. El texto de entrada combina el título y la sinopsis de la película.

- **Métrica de referencia:** ROC AUC macro ≈ 0.899 en validación
- **Géneros predichos (24):** Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, History, Horror, Music, Musical, Mystery, News, Romance, Sci-Fi, Short, Sport, Thriller, War, Western

---

## 🚀 Despliegue en Render

La API está disponible en:

```
https://proyecto2-genero-peliculas.onrender.com
```

### Flujo de despliegue automático

```
git push → Render detecta cambios
         → pip install -r requirements.txt
         → python train.py     # descarga datos, entrena y guarda model_pipeline.pkl
         → uvicorn app:app ... # levanta la API con el modelo cargado
```

---

## 🔌 Endpoints

| Método | Ruta             | Descripción                                   |
|--------|------------------|-----------------------------------------------|
| GET    | `/`              | Información general de la API                 |
| GET    | `/health`        | Estado del servicio y modelo                  |
| POST   | `/predict`       | Predicción para una película                  |
| POST   | `/predict_batch` | Predicción para múltiples películas (máx. 100)|
| GET    | `/docs`          | Documentación interactiva Swagger UI          |
| GET    | `/redoc`         | Documentación ReDoc                           |

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
    "p_Action":    0.3437,
    "p_Thriller":  0.3235,
    "p_Drama":     0.3214,
    "p_Adventure": 0.2286,
    "...": "..."
  },
  "top_genres": ["Action", "Thriller", "Drama"]
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
      "plot": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.",
      "year": 1995
    },
    {
      "title": "Psycho",
      "plot": "A secretary embezzles money from her employer's client, goes on the run, and checks into a remote motel run by a young man under the domination of his mother.",
      "year": 1960
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
      "probabilities": { "p_Animation": 0.72, "p_Comedy": 0.61, "..." : "..." },
      "top_genres": ["Animation", "Comedy", "Family"]
    },
    {
      "title": "Psycho",
      "probabilities": { "p_Horror": 0.68, "p_Thriller": 0.65, "...": "..." },
      "top_genres": ["Horror", "Thriller", "Drama"]
    }
  ],
  "total_movies": 2,
  "processing_time_seconds": 0.032
}
```

---

## ⚙️ ¿Cómo funciona la aplicación?

La aplicación tiene dos componentes principales: **`train.py`** y **`app.py`**. Así es el flujo completo:

### 1. Entrenamiento del modelo (`train.py`)

Este script se ejecuta **una sola vez**, durante el despliegue en Render (o manualmente en local). Su trabajo es:

```
1. Descarga el dataset de películas (dataTraining.zip) desde GitHub
2. Carga los datos en un DataFrame de pandas
3. Combina título + sinopsis de cada película en un solo texto
4. Entrena el pipeline:
      TF-IDF  →  convierte el texto en números (vectores)
      Regresión Logística One-vs-Rest  →  aprende a predecir los 24 géneros
5. Guarda todo el modelo entrenado en el archivo "model_pipeline.pkl"
```

> **¿Qué es `model_pipeline.pkl`?**  
> Es el modelo ya entrenado, guardado en disco. Contiene el vectorizador TF-IDF y el clasificador. Una vez creado, la API lo usa directamente sin necesidad de volver a entrenar.

---

### 2. Servidor de predicciones (`app.py`)

Cuando la API arranca (al iniciar el servidor), **carga automáticamente** el archivo `model_pipeline.pkl` en memoria. Esto ocurre una sola vez al inicio gracias a la función `lifespan`:

```
Servidor arranca
     ↓
Carga "model_pipeline.pkl" → queda en memoria RAM
     ↓
API lista para recibir peticiones
     ↓
POST /predict   →  toma título + sinopsis
               →  el modelo calcula probabilidades para 24 géneros
               →  retorna JSON con probabilidades y top 3 géneros
```

**Resumen visual del flujo completo:**

```
[Despliegue]                          [Uso en producción]
train.py                              app.py (servidor activo)
   │                                       │
   ├─ Descarga datos                       ├─ Carga model_pipeline.pkl al inicio
   ├─ Entrena TF-IDF + LogReg             │
   └─ Guarda model_pipeline.pkl ─────────►│
                                          ├─ POST /predict
                                          │    ├─ texto = título + sinopsis
                                          │    ├─ modelo calcula probabilidades
                                          │    └─ responde JSON con géneros
                                          │
                                          └─ POST /predict_batch (hasta 100 películas)
```

---

## 🛠️ Ejecución local

Sigue estos pasos para correr la API en tu computador:

```bash
# 1. Clonar el repositorio
git clone https://github.com/MiguelData2030/Proyecto2_Genero_Peliculas.git
cd Proyecto2_Genero_Peliculas

# 2. (Recomendado) Crear entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Entrenar el modelo (descarga datos y genera model_pipeline.pkl)
#    ⚠️  Requiere conexión a internet. Tarda 1-3 minutos.
python train.py

# 5. Levantar la API
uvicorn app:app --reload

# 6. Abrir la documentación interactiva en el navegador
#    http://127.0.0.1:8000/docs
```

> **Nota:** El paso 4 (`python train.py`) solo es necesario la primera vez o si quieres reentrenar el modelo. Una vez generado `model_pipeline.pkl`, puedes ir directo al paso 5.

---

## 🗂️ Estructura del proyecto

```
Proyecto2_Genero_Peliculas/
├── app.py                  # API FastAPI (endpoints e inferencia)
├── train.py                # Entrenamiento del modelo y serialización
├── requirements.txt        # Dependencias Python
├── render.yaml             # Configuración despliegue Render.com
├── .python-version         # Versión de Python para Render
├── README.md               # Documentación
└── .gitignore
```

---

## 🔧 Arquitectura del modelo

```
Entrada (título + sinopsis)
       ↓
requests.get() — descarga dataTraining.zip directamente
       ↓
TF-IDF Vectorizer
  • max_features = 20 000
  • ngram_range  = (1, 2)
  • min_df = 2  |  max_df = 0.95
  • sublinear_tf = True
  • stop_words = "english"
       ↓
OneVsRestClassifier
  └── LogisticRegression (liblinear, C=1.0, n_jobs=1)
       ↓
Probabilidades para 24 géneros
```

---

## 👥 Integrantes

| Nombre |
|--------|
| Adolfo Ramírez Moreno |
| Gisell Zarina Gutiérrez Fernandez |
| Miguel Ángel Londoño Díaz |
| Winston Andrés Licona Briceño |

**Proyecto 2 — MIAD Uniandes**
