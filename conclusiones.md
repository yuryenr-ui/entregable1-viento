# Conclusiones y limitaciones

## 9.1 Hallazgos

1. **Los datos.** Con las mediciones horarias del IDEAM construimos un panel de 1.3 millones de estación-horas en
   76 estaciones del Caribe colombiano (2020–2025), con una limpieza que no usa información futura. El viento tiene
   ciclos diario y anual fuertes, un gradiente espacial marcado y una dependencia temporal y espacial que obliga a
   validar por periodos (capítulo 1).
2. **Los modelos del curso tienen un techo común.** Con validación anidada temporal y cuatro optimizadores, los
   siete modelos quedan muy cerca: en 2025 el RMSE va de `0.7227` (Random Forest) a `0.7607 m/s` (árbol), Random
   Forest y XGBoost no se distinguen y sus errores se correlacionan entre `0.976` y `0.986`. El límite lo pone la
   información disponible más que el algoritmo (capítulo 5).
3. **El modelo original.** Proponemos un ensamble que promedia sin pesos Random Forest, XGBoost, SVR y dos
   versiones de la regresión por diferencias de pares por estación. PADRE por estación aprende cuánto se aparta una
   hora de otras horas conocidas de la misma estación: el efecto fijo del sitio se cancela en la diferencia y vuelve
   a entrar por las anclas. Sus errores se correlacionan menos con los del curso (`0.92`–`0.93`) y, por la
   descomposición de ambigüedad, ese desacuerdo es lo que baja el error del promedio (capítulo 6).
4. **Es mejor, con una ventaja pequeña.** El ensamble tiene el menor RMSE medio del bucle externo (`0.8826`, frente
   a `0.8961` de Random Forest) y de 2025 (`0.7193`), y supera a cada uno de los siete modelos con
   Diebold-Mariano, Giacomini-White y Clark-West con ajuste de Holm, el MCS y el SPA. La ventaja es estable ante
   cinco semillas, los cuatro optimizadores y los hiperparámetros propios, pero es de `1.5 %` del RMSE en el bucle
   externo y `0.5 %` en 2025 (capítulo 8).
5. **Lo que no cambia.** Ningún modelo, tampoco el propuesto, corrige el sesgo en vientos fuertes, y los residuos
   siguen siendo dependientes en todas las estaciones (sección 8.4).

## 9.2 Veredicto sobre las tres condiciones del modelo original

| Condición | Evidencia | Veredicto |
|---|---|---|
| **Novedad en la aplicación (R1):** el modelo no se ha usado para pronosticar viento | Cinco bases y ocho cadenas sin aplicaciones de PADRE a viento; comparación con nueve métodos afines de otras familias (capítulo 7) | Se cumple, con el alcance de la búsqueda descrita |
| **Implementación algorítmica propia (R2):** no proviene de un *import* | Clases propias compatibles con scikit-learn en `e3/modelo_original.py`, pseudocódigo, complejidad teórica contrastada con tiempos medidos y pruebas unitarias con pytest (sección 6.1). El motor que aprende la diferencia es un Random Forest o un XGBoost de biblioteca, usado sin modificar; lo propio es el algoritmo que lo envuelve: la construcción de pares por estación, la regla de la vecina, el sorteo y el promedio de anclas y el ensamble | Se cumple. El núcleo nuevo es el meta-algoritmo, no el motor |
| **Superioridad estadística (R3):** mejor que los modelos del curso con al menos tres pruebas significativas y efecto no despreciable | Menor RMSE medio del bucle externo que los siete; cinco procedimientos significativos con Holm en 2024 y en 2025; `|d de Cohen|` de `0.25` a `0.79` en el bucle externo (capítulo 8) | Se cumple en el bucle externo. En 2025 la diferencia con Random Forest es significativa pero de efecto pequeño (`d = 0.18`) |

## 9.3 Limitaciones y trabajo futuro

- **Un solo problema y un solo año de prueba.** Evaluamos el modelo en un único conjunto de datos y con 2025 como
  prueba. No lo probamos en otros conjuntos de regresión ni frente a métodos recientes fuera de los del curso,
  así que no podemos afirmar que la ventaja se generalice a otros problemas.
- **Supuesto de nivel constante.** PADRE supone que el efecto de cada estación no cambia. Cuando cambia, como en
  Puerto Estrella, las anclas arrastran el nivel antiguo; usar anclas recientes sería la corrección natural, aunque
  en 2024 las de los últimos `90` días dieron peor resultado.
- **Sensibilidad al ruido.** Con ruido en el objetivo PADRE se degrada más que XGBoost, porque el ruido entra en la
  diferencia y en las anclas (sección 8.5).
- **Costo.** El ensamble tarda unos diez minutos en ajustarse, cuatro veces lo que Random Forest, y predecir cuesta
  `20` evaluaciones del motor por fila.
- **Vientos fuertes y dependencia.** Para corregirlos hace falta información sobre el estado futuro de la
  atmósfera. El paso siguiente con más potencial es agregar predictoras de pronósticos numéricos (GFS) o de
  reanálisis (ERA5) disponibles al momento de emitir el pronóstico, y probar PADRE con anclas ponderadas por
  cercanía en el tiempo.

## 9.4 Declaración de autoría y uso de herramientas de inteligencia artificial

Este trabajo fue realizado por Yuryen Rollo y Betzaida Ruiz. Definimos el problema, escogimos los datos, tomamos
las decisiones metodológicas (partición, validación, reglas fijadas antes de evaluar, criterios de éxito) y
revisamos los resultados.

Usamos Claude (Anthropic), mediante Claude Code, como asistente de programación y de análisis: para escribir y
depurar el código de los scripts y de los módulos de `e3/`, incluido el modelo original y sus pruebas unitarias,
para ejecutar los experimentos, para buscar y resumir literatura y para redactar borradores del texto del libro.
Revisamos el código y el texto, verificamos que los resultados se reproducen con los scripts del repositorio y
somos responsables de cada línea entregada.

*Firmado: Yuryen Rollo, Betzaida Ruiz.*
