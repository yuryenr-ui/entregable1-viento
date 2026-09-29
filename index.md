# Pronóstico de la velocidad del viento a 24 h en el Caribe colombiano

**Curso:** Machine Learning · **Profesor:** Lihki Rubio Ortega · **Grupo:** Yuryen Rollo, Betzaida Ruiz

Este libro contiene el **Entregable 1** del proyecto de investigación: selección de la base de datos,
análisis exploratorio (EDA) y modelo base.

- **Datos:** estaciones automáticas del IDEAM publicadas en el Portal Nacional de Datos Abiertos
  (datos.gov.co): velocidad y dirección del viento, temperatura del aire y presión atmosférica,
  7 departamentos del Caribe, 2020–2025. Licencia: CC BY-SA 4.0, atribución al IDEAM.
- **Problema:** pronosticar la velocidad media horaria del viento 24 horas después de la última hora
  observada.
- **Modelo base:** SVR lineal, comparado con persistencia, climatología y un modelo de media.

## Reproducibilidad

1. Descargar los CSV de datos.gov.co (enlaces en la sección 1.3 del entregable), uno por variable y año,
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

**Nota metodológica:** el EDA usa la depuración original, que es retrospectiva (control de calidad con
información de la estación-año completa). Todo el modelado (sección 3) usa una
limpieza causal (`--causal`), en la que cada lectura se limpia solo con información anterior a ella.

Límites departamentales del mapa base: [geoBoundaries](https://www.geoboundaries.org) (fuente
OpenStreetMap, licencia ODbL 1.0).
