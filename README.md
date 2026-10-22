# Pronóstico de la velocidad del viento a 24 h en el Caribe colombiano

Proyecto del curso de Machine Learning (Universidad del Norte). Grupo: Yuryen Rollo y Betzaida Ruiz.

Pronosticamos la velocidad media del viento de la hora que empieza 24 horas después de la última hora observada,
en cada estación automática del IDEAM del Caribe colombiano, con datos de 2020 a 2025. El trabajo incluye el ETL y
el análisis exploratorio, la comparación de siete modelos de regresión con cuatro optimizadores de hiperparámetros
bajo validación anidada temporal, y un modelo original: un ensamble con regresión por diferencias de pares por
estación (PADRE) que supera con diferencia significativa a los siete modelos.

- **Libro (Jupyter Book):** https://yuryenr-ui.github.io/entregable1-viento/
- **Dashboard:** https://viento-caribe-dashboard.onrender.com (código en https://github.com/yuryenr-ui/entregable2-viento)

## Instalación

```bash
conda create -n ml_env python=3.11
conda activate ml_env
pip install -r requirements.txt
```

XGBoost usa la GPU (`device="cuda"`) cuando está disponible. La semilla global es `42` (`e3/__init__.py`).

## Uso

El orden completo de ejecución, con tiempos aproximados, está en el capítulo 10 del libro
(`reproducibilidad.ipynb`). Lo esencial:

```bash
python scripts/e3_preparar_datos.py      # conjunto de modelado
python scripts/e3_experimentos.py        # 7 modelos × 4 optimizadores con validación anidada
python scripts/ensamble_padre.py         # modelo original en 2024 y 2025
python scripts/padre_tabla_pruebas.py    # pruebas de hipótesis del modelo original
python scripts/padre_final.py            # reentrenamiento final, artefactos y SHAP
python -m pytest tests -q                # pruebas unitarias del modelo original
```

El modelo original se usa como cualquier estimador de scikit-learn; la primera columna de `X` es el código de la
estación:

```python
from e3.modelo_original import PadreEstacionRegresor
from xgboost import XGBRegressor

modelo = PadreEstacionRegresor(XGBRegressor(), pares=4, anclas=20, coords=coordenadas)
modelo.fit(X_entrenamiento, y_entrenamiento)
pronostico = modelo.predict(X_nuevo)
```

## Estructura

| Carpeta o archivo | Contenido |
|---|---|
| `index.md`, `*.py` / `*.ipynb`, `*.md` | Capítulos del libro (los `.ipynb` se generan desde los `.py` con `a_notebook.py`) |
| `e3/` | Módulos: datos, modelos, optimizadores, evaluación estadística y el modelo original (`modelo_original.py`) |
| `scripts/` | Experimentos y procesamiento de datos |
| `tests/` | Pruebas unitarias del modelo original (pytest) |
| `artifacts/` | Modelos serializados con joblib y su `metadata.json` |
| `resultados/` | Resultados versionados: predicciones, tablas y búsquedas |
| `references.bib` | Bibliografía |
| `myst.yml` | Configuración del libro (Jupyter Book 2) |

## Publicación del libro

Cada `push` a `main` construye el libro con GitHub Actions y lo publica en GitHub Pages. Para construirlo en local:
`jupyter-book build --html` (requiere Node.js).

## Licencia

Datos del IDEAM bajo CC BY-SA 4.0; este trabajo se publica con la misma licencia.
