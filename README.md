# Clasificación de tumores de mama con redes neuronales (MLP y Transformer) y base de datos SQL

Proyecto ACA N.° 2 – **Aprendizaje Automático** · CUN · Especialización en Inteligencia Artificial
**Autor:** Cristian Fernando Marín Caro · **Docente:** Hamilton Rivera Flor · **Pilar:** Salud · **Dataset:** [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

Se comparan dos redes neuronales profundas para clasificar tumores como benignos o malignos a partir de 30 medidas del núcleo celular, con el dataset gestionado en una base **SQLite** y una interfaz en **Streamlit**:

* **MLP profunda:** `Entrada → GaussianNoise → [Dense → BatchNorm → ReLU → Dropout] ×3 (128, 64, 32) → Sigmoide`
* **FT-Transformer** (Gorishniy et al., 2021): cada variable se convierte en un *token*, se agrega un token `[CLS]` y 2 bloques Transformer (autoatención de 4 cabezas) los combinan.

## Flujo metodológico

1. **Exploración y gestión SQL:** el dataset se guarda en `wdbc.db` (clave primaria, restricciones `CHECK`, índices) y se explora con consultas SQL.
2. **Preparación:** partición estratificada 70/15/15 registrada en SQL, estandarización con estadísticas de entrenamiento, aumento de datos (ruido gaussiano) y pesos de clase.
3. **Modelado:** entrenamiento con Adam, parada temprana y reducción de la tasa de aprendizaje.
4. **Evaluación:** curvas de pérdida y exactitud, accuracy, precision, recall, F1, ROC-AUC, matriz de confusión, ajuste del umbral, estabilidad con 5 semillas e importancia de variables.
5. **Despliegue:** app de Streamlit con predicción de ambas redes, explorador y consola SQL de solo lectura, y panel de métricas.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `ACA2_Aprendizaje_Automatico_Redes_Neuronales.ipynb` | Notebook reproducible: SQL, preprocesamiento, entrenamiento, evaluación y exportación |
| `app.py` | Interfaz Streamlit |
| `modelos_dl.py` | Arquitecturas y utilidades (compartido por el notebook y la app) |
| `requirements.txt` | Dependencias de la app |
| `MLP_profunda.keras`, `FT_Transformer.keras` | Modelos entrenados |
| `escalador.json`, `metricas.json`, `wdbc.db` | Estandarización, métricas del entrenamiento y base SQL |

## Cómo ejecutarlo

1. Abre el notebook en **Google Colab** y ejecútalo con *Entorno de ejecución → Ejecutar todo* (unos 3 a 6 minutos en CPU; no requiere GPU). Al final descarga `artefactos_streamlit.zip`.
2. Copia los archivos del zip (`*.keras`, `escalador.json`, `metricas.json`, `wdbc.db`) junto a `app.py`.
3. Local:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. **Streamlit Community Cloud:** en *Advanced settings* elige **Python 3.12**. La app no necesita claves ni secretos.

## Notas técnicas

* Los modelos se entrenan con el backend TensorFlow (el de Colab) y la app los carga con **JAX**, que es más liviano; Keras 3 guarda los modelos en un formato independiente del backend (se comprobó que las predicciones coinciden).
* La predicción llama al modelo directamente y no usa `model.predict`, porque `predict` con JAX no es seguro entre hilos y Streamlit ejecuta cada interacción en un hilo distinto.
* La consola SQL de la app solo acepta una sentencia `SELECT` y abre la base en modo solo lectura.

## Limitaciones

Dataset pequeño (569 casos) y de una sola fuente; conjunto de prueba de 86 pacientes, por lo que las métricas varían de una corrida a otra (se reporta la media y la desviación con 5 semillas); sin validación externa.

## Aviso

Prototipo académico. No ofrece diagnóstico médico ni reemplaza el criterio de un profesional de la salud.

## Referencias

Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems, 34*, 18932–18943. https://arxiv.org/abs/2106.11959

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*, 5998–6008. https://arxiv.org/abs/1706.03762

Wolberg, W., Street, N., & Mangasarian, O. (1995). *UCI Breast Cancer Wisconsin (Diagnostic)* [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
