# Pronóstico de la velocidad del viento a 24 h en el Caribe colombiano

**Curso:** Machine Learning · **Profesor:** Lihki Rubio Ortega · **Grupo:** Yuryen Rollo, Betzaida Ruiz

¿Se puede anticipar con un día de antelación la velocidad del viento que medirá una estación del
Caribe colombiano? En este trabajo lo estudiamos con las mediciones horarias del IDEAM: describimos y
depuramos los datos, analizamos su comportamiento en el tiempo y en el espacio, comparamos siete modelos de
regresión con cuatro optimizadores y proponemos un modelo propio que los supera.

## Resumen ejecutivo

- **Problema:** pronosticar la velocidad media del viento en la hora que empieza 24 horas después de la última
  hora observada, en cada estación, con lo medido hasta ese momento. Sirve para planear la operación del día
  siguiente en parques eólicos, puertos y aeropuertos de la región.
- **Datos:** estaciones automáticas del IDEAM publicadas en el Portal Nacional de Datos Abiertos
  (datos.gov.co): velocidad y dirección del viento, temperatura del aire y presión atmosférica en
  7 departamentos del Caribe, 2020–2025. Licencia CC BY-SA 4.0, con atribución al IDEAM.
- **Validación:** entrenamiento 2020–2024, validación anidada con los cuatro trimestres de 2024 como bucle externo
  y 2025 como prueba evaluada una sola vez.
- **Resultado:** el modelo original, un ensamble con regresión por diferencias de pares por estación (PADRE), tiene
  el menor error del bucle externo y de 2025 (`RMSE = 0.7193 m/s`, `R² = 0.747`) y supera con diferencia
  significativa a cada uno de los siete modelos del curso. La ventaja es pequeña (`0.5 %` frente a Random Forest
  en 2025) y ningún modelo corrige la subestimación de los vientos fuertes.

## Objetivos

1. Construir, sin fuga de información, un conjunto de datos horario del viento en el Caribe colombiano y
   describir su comportamiento temporal, espacial y espacio-temporal.
2. Comparar siete modelos de regresión con cuatro métodos de optimización de hiperparámetros bajo validación
   anidada temporal y pruebas estadísticas de comparación de pronósticos.
3. Proponer, implementar y validar estadísticamente un modelo original que supere a esos siete modelos.

## Mapa del libro

El libro sigue los tres entregables del curso:

- **Entregable 1, datos, EDA y modelo base** (capítulos 1–4): un SVR lineal, comparado con la persistencia, la
  climatología y un modelo de media. Reduce el error de la persistencia en un `13 %` (`R² = 0.74` en 2025).
- **Entregable 1.5, dashboard:** el tablero interactivo en https://viento-caribe-dashboard.onrender.com.
- **Entregable 2, comparación de modelos** (capítulo 5): siete modelos de regresión con cuatro optimizadores de
  hiperparámetros y validación anidada temporal. El mejor en 2025 es Random Forest (`RMSE = 0.7227 m/s`).
- **Entregable 3, modelo original:** la trazabilidad de correcciones; el modelo propuesto, su algoritmo,
  complejidad, implementación y pruebas (capítulo 6); la revisión de literatura y la declaración de novedad
  (capítulo 7); la validación estadística (capítulo 8); las conclusiones, con el veredicto sobre el modelo y la
  declaración de uso de herramientas de IA (capítulo 9), y la reproducibilidad y la bibliografía (capítulo 10).

## Reproducibilidad

1. Descargar los CSV de datos.gov.co (los enlaces están en la sección 1.3), uno por variable y año,
   en subcarpetas `VELOCIDAD DEL VIENTO`, `DIRECCION DEL VIENTO`, `TEMPERATURA`, `PRESION` y
   `CATALOGO estaciones`.
2. Generar el panel en la carpeta hermana `ideam_viento/`, que es donde lo busca el notebook (los
   scripts escriben en la carpeta desde la que se ejecutan):

   ```
   mkdir ../ideam_viento
   cd ../ideam_viento
   python ../entregable1/scripts/procesar_todo.py "<carpeta de descargas>"
   python ../entregable1/scripts/procesar_todo.py "<carpeta de descargas>" --causal   # panel del modelado y sensibilidad
   python ../entregable1/scripts/verificar_conteo.py "<carpeta de descargas>"
   ```

   Para usar otra carpeta, definir la variable de entorno `DATOS_PANEL` con su ruta.
3. Desde la carpeta del repositorio, `python scripts/revisar_umbral.py "<carpeta VELOCIDAD DEL VIENTO>"`
   genera `resultados/umbral_velocidad.csv`.
4. Ejecutar `entregable1.ipynb` con las versiones de `requirements.txt` (semilla fija: 42).

El análisis exploratorio usa una depuración retrospectiva, que revisa cada estación-año completa. Para el
modelado (sección 3) reprocesamos los datos con una limpieza causal (`--causal`), en la que cada lectura se
limpia solo con información anterior a ella.

Límites departamentales del mapa base: [geoBoundaries](https://www.geoboundaries.org) (fuente
OpenStreetMap, licencia ODbL 1.0).
