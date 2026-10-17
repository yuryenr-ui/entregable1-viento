# Dashboard interactivo

Para presentar los resultados construimos un dashboard con Dash y Plotly, publicado en Render:

**https://viento-caribe-dashboard.onrender.com**

El servidor gratuito se suspende cuando nadie lo usa, así que la primera visita puede tardar cerca de un minuto en
cargar. El código está en https://github.com/yuryenr-ui/entregable2-viento.

## Qué muestra cada pestaña

El tablero reúne las gráficas más importantes del análisis, con filtros para explorarlas. La interpretación
detallada de cada resultado, las pruebas estadísticas y las tablas completas están en las secciones de este libro
que indica la última columna.

| Pestaña | Contenido | Interacción | Dónde se interpreta |
|---|---|---|---|
| Contexto | El problema y por qué importa, cifras del conjunto de datos, mapa de las 76 estaciones, descripción de la fuente y de la limpieza | Mapa con zoom y detalle por estación | Secciones 1.1 a 1.6 |
| EDA | Distribución de la velocidad, ciclos diario y anual | Filtro por departamento | 2.1 y 2.6 |
| EDA | Serie diaria de cada estación (2020–2024) | Selector de estación y barra de zoom | 2.6 |
| EDA | Rosa de los vientos, correlación cruzada, correlaciones de Spearman, asociación de las predictoras con el objetivo | — | 2.2, 2.3 y 2.6 |
| EDA | Mapa espacial: velocidad media, agrupamientos LISA, regímenes de viento y diferencia entre temporadas; semivariograma y correlación según la distancia | Selector del color del mapa | 2.4, 2.7 y 2.8 |
| EDA | Temporada seca frente a lluviosa, velocidad mensual por departamento, descomposición STL, anomalías y ENSO, faltantes por mes, PCA | — | 1.6, 2.4, 2.6 y 2.8 |
| Modelo base | Indicadores con intervalos de confianza y tabla del SVR frente a la persistencia, la climatología y la media | — | 3.2 |
| Modelo base | Mejora sobre la persistencia y su estabilidad en `GroupKFold` y en bloques geográficos | — | 3.2 |
| Modelo base | Pronóstico frente a lo observado en 2025 | Selector de estación y de mes | 3.2 |
| Modelo base | R² por estación (dispersión y mapa), validación cruzada por trimestres, curva de aprendizaje, error según la intensidad del viento, ACF de los residuos y coeficientes | — | 3.1 a 3.5 |

Todas las cifras del tablero coinciden con las de este libro: `R² = 0.737`, `RMSE = 0.733 m/s` y una reducción del
`13.2 %` del error de la persistencia en 2025.

## Cómo está construido

El análisis pesado (limpieza, predictoras, validación y bootstrap) no se repite en el servidor. El script
`scripts/exportar_datos.py` ejecuta el código de este libro, con la misma semilla (`42`) y las mismas particiones, y
guarda unas 50 tablas pequeñas (2.5 MB en total, incluidas las 213 972 predicciones horarias de 2025). Así el tablero
carga rápido y no necesita el panel completo, que no cabría en la memoria de un servidor gratuito.

* `src/figuras.py` carga esas tablas y construye cada gráfica con Plotly, con una plantilla oscura común.
* `src/app.py` arma las tres pestañas con `dash-bootstrap-components` y conecta los menús con las gráficas mediante
  cinco *callbacks*: departamento → distribución y ciclos; departamento → lista de estaciones; estación → serie de
  tiempo; selector → color del mapa espacial; estación y mes → pronóstico frente a lo observado.
* `src/assets/` contiene la hoja de estilos y una ilustración propia en SVG.

El tablero no entrena ningún modelo: muestra las predicciones del SVR ajustado con 2020–2024, con hiperparámetros
elegidos por validación cruzada cronológica en 2024 y evaluado una sola vez en 2025, de modo que no introduce fuga ni
sobreajuste.

**Despliegue.** `render.yaml` le indica a Render cómo instalar las dependencias (`requirements.txt`) y cómo arrancar
el servidor (`gunicorn --chdir src app:server`, gracias a la línea `server = app.server` de la aplicación).

**Ejecutarlo en el computador:**

```
pip install -r requirements.txt
python src/app.py
```

y abrir http://127.0.0.1:8050/.
