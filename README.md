# challenge-tecnico
Este proyecto consiste en la aplicación de técnicas de aprendizaje automático para resolver un problema de clasificación utilizando un dataset proporcionado (`train.csv`).

 Estructura del Proyecto

El análisis se desarrolla en un notebook dividido en las siguientes secciones:

1. Carga y Exploración Inicial de Datos
- Revisión general de la estructura del dataset.
- Identificación de valores faltantes y posibles outliers.

2. Análisis Exploratorio de Datos (EDA)
- Estadísticas descriptivas básicas.
- Visualizaciones para entender la distribución de las variables y sus relaciones.
- Limpieza de valores nulos y tratamiento de outliers.

3. Preprocesamiento de Datos
- Codificación de variables categóricas (One-Hot Encoding).
- Escalado de variables numéricas.
- División en conjuntos de entrenamiento y prueba.

4. Implementación de Modelos y Benchmark
- Entrenamiento y evaluación inicial de cinco modelos:
  - Regresión Logística
  - K-Nearest Neighbors (KNN)
  - Árbol de Decisión
  - XGBoost
  - LightGBM
- Evaluación mediante validación cruzada (`cross_val_score`).

5. Optimización de Hiperparámetros
- Optimización del modelo LightGBM utilizando `GridSearchCV`.

6. Comparación de Modelos
- Evaluación final del mejor modelo usando métricas como:
  - Accuracy
  - F1 Score
  - ROC-AUC
  - Reporte de clasificación

 Requisitos
- Python 3.8+
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn
- XGBoost
- LightGBM

> El notebook está preparado para ejecutarse en Google Colab.
