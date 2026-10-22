# Trazabilidad de correcciones

Esta tabla recoge las observaciones recibidas sobre las entregas anteriores, qué hicimos con cada una, dónde
está en este libro y con qué evidencia. Cuando una corrección cambió resultados, recalculamos todas las tablas y
figuras que dependían de ellos.

## Observaciones recibidas

La única observación con pedido de cambio llegó con la calificación del primer capítulo (datos, EDA y modelo
base). Sobre la comparación de modelos no recibimos observaciones antes del cierre de este libro.

| Observación | Acción realizada | Ubicación | Evidencia | Resultado |
|---|---|---|---|---|
| Intentar corregir, con un modelo no lineal, el sesgo del modelo base en vientos fuertes | Comparamos siete modelos, cinco de ellos no lineales (KNN, árbol, Random Forest, XGBoost y SVR con núcleo RBF), y propusimos un ensamble no lineal. Medimos el sesgo por rango de velocidad observada y pronosticada y en el `5 %` de horas más ventosas | Sección 8.4 | Tablas de sesgo y RMSE por rango de velocidad (2025) | **No corregido.** En las horas con más de `4.95 m/s` el sesgo del modelo lineal es `−1.08 m/s` y el del ensamble `−1.20 m/s`. Ningún modelo lo reduce; la ganancia de los no lineales está en los vientos bajos y moderados |
| Intentar corregir la dependencia que quedó en los residuos | Mismos modelos; Ljung-Box (24 rezagos) y BDS por estación y autocorrelación media de los residuos | Secciones 8.4 y 5.4 (diagnóstico de residuos) | Tabla de dependencia de los residuos por modelo | **Parcial.** La autocorrelación a 24 h baja de `−0.111` (lineal) a `−0.065` (ensamble), pero Ljung-Box y BDS siguen rechazando la independencia en las `53` estaciones |

**Por qué no se corrigió del todo.** Un pronóstico que minimiza el error cuadrático predice la media condicional:
si nada en lo medido hasta hoy anuncia que mañana el viento se duplicará, ningún algoritmo puede anticiparlo, y
los extremos quedan subestimados. Lo mismo vale para la dependencia de los residuos: los errores de horas
consecutivas se parecen porque comparten el mismo cambio de régimen que las predictoras no ven. Los análisis de
SHAP y la curva de aprendizaje de las secciones 5.5 y 5.6 apuntan a lo mismo: el límite lo pone la información disponible.
Corregirlo requiere predictoras sobre el estado futuro de la atmósfera (pronósticos numéricos como GFS o
reanálisis como ERA5), que dejamos como trabajo futuro (sección 9.3).

## Revisiones propias

Además de lo anterior, después de entregar el primer capítulo hicimos estos cambios por revisión propia:

| Hallazgo propio | Acción realizada | Ubicación | Efecto en los resultados |
|---|---|---|---|
| La sección 3.7 y las conclusiones repetían resultados ya presentados | Se quitaron las repeticiones | Secciones 3.7 y 4 | Ninguno |
| Las conclusiones mencionaban redes neuronales como paso siguiente, fuera del alcance del proyecto | Se quitó la mención | Sección 4 | Ninguno |
