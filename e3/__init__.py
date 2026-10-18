"""Módulos de la sección 4: comparación de modelos con optimización de hiperparámetros y validación anidada.

- ``datos``: carga del conjunto de modelado y pliegues cronológicos (externos e internos).
- ``modelos``: definición de los 7 modelos de regresión y de sus espacios de búsqueda.
- ``optimizadores``: búsqueda en rejilla, aleatoria, bayesiana (Optuna), genética (DEAP) y eliminación sucesiva.
- ``evaluacion``: métricas, diagnóstico de residuos y pruebas estadísticas de comparación.
"""
SEED = 42  # semilla global única: numpy, scikit-learn, XGBoost, Optuna y DEAP
