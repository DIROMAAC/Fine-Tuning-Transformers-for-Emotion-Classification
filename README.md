# Detección de Emociones en Texto (Text Emotion Detection)

Este proyecto está diseñado para clasificar y detectar emociones en textos en inglés utilizando modelos de Procesamiento de Lenguaje Natural (NLP) avanzados. Implementa dos enfoques complementarios: el uso de un modelo preentrenado para inferencia por lotes (batch inference) y el ajuste fino (fine-tuning) de una arquitectura basada en Transformers (`distilroberta-base`) con un dataset de gran escala (~416,809 registros).

---

## 📋 Tabla de Contenidos
- [Descripción General](#descripción-general)
- [Mapeo de Emociones](#mapeo-de-emociones)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Dataset y Análisis Exploratorio](#dataset-y-análisis-exploratorio)
- [Configuración del Entorno de Desarrollo](#configuración-del-entorno-de-desarrollo)
- [Guía de Uso](#guía-de-uso)
  - [Inferencia por Lotes](#inferencia-por-lotes)
  - [Ajuste Fino del Modelo](#ajuste-fino-del-modelo)
- [Evaluación y Resultados](#evaluación-y-resultados)
- [Requisitos de Hardware y Aceleración](#requisitos-de-hardware-y-aceleración)

---

## 🔍 Descripción General

El objetivo principal es identificar una de las seis emociones básicas en textos. El proyecto aprovecha el ecosistema de **Hugging Face** (`transformers`, `datasets`, `trainer`) y la biblioteca **PyTorch** con soporte CUDA para acelerar el procesamiento de datos y entrenamiento a través de la GPU.

Se presentan dos flujos de trabajo principales en formato Jupyter Notebook:
1. **Inferencia por Lotes (`Project.ipynb`)**: Carga el dataset original y utiliza una canalización (`pipeline`) optimizada de clasificación de texto para predecir las emociones utilizando el modelo preentrenado `bhadresh-savani/bert-base-uncased-emotion`.
2. **Entrenamiento y Ajuste Fino (`untrained.ipynb`)**: Realiza el entrenamiento de un clasificador de secuencia personalizado a partir de `distilroberta-base` usando el dataset del proyecto, logrando un rendimiento superior de clasificación.

---

## 🏷️ Mapeo de Emociones

El dataset etiqueta los textos con valores numéricos del `0` al `5`, los cuales corresponden a las siguientes emociones:

| ID | Emoción (Inglés) | Emoción (Español) |
|:--:|:----------------:|:-----------------:|
| 0  | sadness          | Tristeza          |
| 1  | joy              | Alegría           |
| 2  | love             | Amor              |
| 3  | anger            | Ira / Enojo       |
| 4  | fear             | Miedo             |
| 5  | surprise         | Sorpresa          |

---

## 📂 Estructura del Proyecto

A continuación se describen los archivos clave que componen el repositorio:

*   **`Project.ipynb`**: Notebook de análisis y clasificación con modelo preentrenado. Contiene el análisis exploratorio visual del dataset y el script de predicción en lotes usando la GPU.
*   **`untrained.ipynb`**: Notebook de entrenamiento. A pesar del nombre, este archivo contiene el pipeline completo para tokenizar, entrenar con la clase `Trainer` de Hugging Face, guardar localmente el modelo ajustado y evaluar sus métricas de rendimiento.
*   **`dataset.csv`**: Dataset original que contiene las columnas `text` (el fragmento de texto) y `label` (el ID numérico de la emoción correspondiente).
*   **`dataset_with_predictions.csv`**: Dataset generado después de ejecutar la inferencia por lotes, añadiendo la etiqueta de predicción final de la emoción.
*   **`Emotion recognition from text2.docx`**: Reporte y documentación teórica detallada sobre el proyecto de detección de emociones.

---

## 📊 Dataset y Análisis Exploratorio

El dataset cuenta con un total de **416,809 registros**, distribuidos de la siguiente manera:

*   **1 (Joy / Alegría):** 141,067 textos
*   **0 (Sadness / Tristeza):** 121,187 textos
*   **3 (Anger / Ira):** 57,317 textos
*   **4 (Fear / Miedo):** 47,712 textos
*   **2 (Love / Amor):** 34,554 textos
*   **5 (Surprise / Sorpresa):** 14,972 textos

En `Project.ipynb` se incluye un gráfico de barras utilizando `seaborn.countplot` para visualizar este balance de clases, permitiendo comprender la distribución del dataset antes del modelado.

---

## ⚙️ Configuración del Entorno de Desarrollo

Para ejecutar este proyecto de forma local, se recomienda crear un entorno virtual de Python. Puedes instalar las librerías necesarias ejecutando los siguientes comandos:

1. **Crear e iniciar el entorno virtual:**
   ```bash
   python -m venv venv
   # En Windows (CMD/PowerShell)
   .\venv\Scripts\activate
   # En macOS/Linux
   source venv/bin/activate
   ```

2. **Instalar dependencias necesarias:**
   Asegúrate de instalar PyTorch con soporte para tu versión de CUDA si cuentas con una GPU compatible (p. ej., NVIDIA RTX).
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers datasets accelerate scikit-learn pandas numpy matplotlib seaborn tqdm notebook
   ```

---

## 🚀 Guía de Uso

### Inferencia por Lotes
Para clasificar los textos usando el modelo ya preentrenado en Hugging Face:
1. Abre `Project.ipynb` en tu entorno de Jupyter.
2. Ejecuta todas las celdas secuencialmente.
3. El script dividirá el conjunto de datos de `dataset.csv` en lotes (`batch_size=64`), procesará el texto a través del pipeline utilizando GPU (`device=0`), y exportará el archivo final `dataset_with_predictions.csv`.

### Ajuste Fino del Modelo
Para entrenar tu propio clasificador basado en `distilroberta-base`:
1. Abre `untrained.ipynb`.
2. Ejecuta las celdas para realizar el tokenizado del texto mediante `AutoTokenizer` y configurar el entrenamiento.
3. El entrenamiento está configurado para ejecutarse durante **3 épocas** con un tamaño de lote de 32, utilizando el optimizador optimizado de Hugging Face a través de `Trainer`.
4. Al finalizar, el modelo y el tokenizador ajustados se guardarán automáticamente en la carpeta local `./emotion_model`.

---

## 📈 Evaluación y Resultados

El modelo entrenado a medida (`distilroberta-base` ajustado en `untrained.ipynb`) se evaluó sobre un conjunto de prueba del 20% (~83,362 textos), obteniendo resultados excelentes:

*   **Exactitud General (Accuracy):** 94.00%
*   **F1-Score Promedio (Weighted Avg):** 94.00%

### Reporte de Clasificación Detallado:

| Clase | Emoción | Precisión | Sensibilidad (Recall) | F1-Score | Soporte (Ejemplos) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | Sadness | 1.00 | 0.96 | 0.98 | 24,240 |
| **1** | Joy | 0.92 | 1.00 | 0.96 | 28,110 |
| **2** | Love | 1.00 | 0.70 | 0.83 | 6,974 |
| **3** | Anger | 0.94 | 0.95 | 0.95 | 11,552 |
| **4** | Fear | 0.85 | 0.99 | 0.91 | 9,475 |
| **5** | Surprise | 0.99 | 0.65 | 0.78 | 3,011 |

*Nota: Se observa un rendimiento excepcional en emociones de mayor volumen como Sadness y Joy, mientras que en Surprise y Love (que tienen menor cantidad de ejemplos en el dataset) el F1-Score se ve ligeramente reducido debido a la sensibilidad (Recall), lo cual es de esperar en datasets desbalanceados.*

---

## 💻 Requisitos de Hardware y Aceleración

Debido al gran volumen de datos (más de 400,000 registros), el procesamiento y entrenamiento mediante CPU es extremadamente lento. Se recomienda encarecidamente contar con:
*   **GPU NVIDIA** compatible con CUDA (p. ej., el entrenamiento se validó exitosamente usando una NVIDIA GeForce RTX 3050 Laptop GPU en aprox. 3 horas y 28 minutos).
*   Mínimo **8 GB de RAM** (16 GB recomendado) para la manipulación eficiente del dataset en memoria.
