# Pronóstico de la velocidad del viento a 24 h en el Caribe colombiano

**Curso:** Machine Learning · **Profesor:** Lihki Rubio Ortega · **Grupo:** Yuryen Rollo, Betzaida Ruiz

¿Se puede anticipar con un día de antelación la velocidad del viento que medirá una estación del
Caribe colombiano? En este trabajo lo estudiamos con las mediciones horarias del IDEAM: describimos y
depuramos los datos, analizamos su comportamiento en el tiempo y en el espacio, y ajustamos un primer
modelo que comparamos con referencias simples.

- **Datos:** estaciones automáticas del IDEAM publicadas en el Portal Nacional de Datos Abiertos
  (datos.gov.co): velocidad y dirección del viento, temperatura del aire y presión atmosférica en
  7 departamentos del Caribe, 2020–2025. Licencia CC BY-SA 4.0, con atribución al IDEAM.
- **Pregunta:** cuál será la velocidad media del viento en la hora que empieza 24 horas después de la
  última hora observada.
- **Modelo base:** un SVR lineal, comparado con la persistencia, la climatología y un modelo de media.
  Reduce el error de la persistencia en un 13 % (R² = 0.74 en 2025).

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
