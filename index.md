# Pronóstico de la velocidad del viento a 24 h en el Caribe colombiano

**Curso:** Machine Learning · **Profesor:** Lihki Rubio Ortega · **Grupo:** Yuryen Rollo, Betzaida Ruiz

Este libro contiene el **Entregable 1** del proyecto de investigación: selección de la base de datos,
análisis exploratorio (EDA) y modelo base.

- **Datos:** estaciones automáticas del IDEAM publicadas en el Portal Nacional de Datos Abiertos
  (datos.gov.co): velocidad y dirección del viento, temperatura del aire y presión atmosférica,
  7 departamentos del Caribe, 2020–2025.
- **Problema:** pronosticar la velocidad media horaria del viento 24 horas después de la última hora
  observada.
- **Modelo base:** SVR lineal, comparado con persistencia, climatología y un modelo de media.

## Reproducibilidad

1. Descargar los CSV de datos.gov.co (enlaces en la sección 1.3 del entregable), uno por variable y año,
   en subcarpetas `VELOCIDAD DEL VIENTO`, `DIRECCION DEL VIENTO`, `TEMPERATURA`, `PRESION` y
   `CATALOGO estaciones`.
2. Generar el panel con `python scripts/procesar_todo.py "<carpeta de descargas>"` (produce
   `panel_multivariado.csv` y los reportes de auditoría). `scripts/verificar_conteo.py` hace un conteo
   independiente sobre los archivos crudos.
3. Ejecutar `entregable1.ipynb` con las versiones de `requirements.txt` (semilla fija: 42).

Límites departamentales del mapa base: [geoBoundaries](https://www.geoboundaries.org) (fuente
OpenStreetMap, licencia ODbL 1.0).
