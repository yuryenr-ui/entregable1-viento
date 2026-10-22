# Revisión de literatura y novedad

En este capítulo documentamos la búsqueda que respalda la novedad del modelo del capítulo 6 y lo comparamos con
los métodos más cercanos de otras familias algorítmicas, para precisar qué es nuevo, qué se conserva y por qué
esos métodos no resuelven el problema que abordamos.

## 7.1 Protocolo de búsqueda

Consultamos cinco bases: **Scopus** y **Web of Science** (con acceso institucional), **OpenAlex** (índice abierto
de más de 250 millones de trabajos, que cubre Crossref, PubMed y repositorios), **Semantic Scholar** y **arXiv**.
Usamos ocho cadenas que combinan (i) el nombre o la familia del modelo (*pairwise difference regression*,
PADRE, *twin regression*, *siamese regression*), (ii) el problema (velocidad o potencia del viento, pronóstico,
estaciones) y (iii) sinónimos en inglés y en español. En OpenAlex buscamos en título y resumen; en Semantic
Scholar, en título y resumen con su sintaxis booleana; en arXiv, en todos los campos. Las consultas de Scopus y
Web of Science se hicieron el 19 de octubre de 2026 y las demás el 22 de octubre de 2026.

Para cada cadena revisamos el título de todos los resultados cuando eran menos de 25, y los 25 primeros por
relevancia cuando eran más; de los que parecían cercanos leímos el resumen y, si hacía falta, el texto completo.

## 7.2 Matriz de novedad

Cadenas (la sintaxis se adaptó a cada base: `AND`/`OR` en OpenAlex, Scopus, Web of Science y arXiv; `+`/`|` en
Semantic Scholar):

| Id | Cadena |
|---|---|
| S0 | `("wind speed" OR "wind power" OR "wind forecast*") AND ("pairwise difference regression" OR "pairwise difference" OR "PADRE")` |
| C1 | `"pairwise difference regression" AND wind` |
| C2 | `"pairwise difference" AND ("wind speed" OR "wind power") AND (forecast OR forecasting OR prediction)` |
| C3 | `PADRE AND ("wind speed" OR "wind power" OR "wind forecast")` |
| C4 | `("regresión por diferencias de pares" OR "diferencias por pares" OR "diferencias de pares") AND (viento OR eólica)` |
| C5 | `("twin neural network regression" OR "twin regression" OR "siamese regression") AND (wind OR meteorology OR weather)` |
| C6 | `("pairwise difference" OR "pairwise regression") AND station AND "wind speed"` |
| C7 | `("pronóstico" OR "predicción") AND "velocidad del viento" AND ("aprendizaje automático" OR "machine learning")` |

| Base de datos | Cadena | Result. | Revis. | Relev. | Observación |
|---|---|---|---|---|---|
| Scopus | S0 | 12 | 12 | 0 | Seis son lugares llamados *Padre* (Padre Island, Padre Las Casas) y seis usan *pairwise differences* para comparar medias de grupos |
| Web of Science | S0 | 65 | 65 | 0 | Tres son lugares llamados *Padre*; los demás no contienen los términos en título, resumen ni palabras clave |
| Semantic Scholar | Artículos que citan a Tynes et al. (2021) | ≈ 58 | 58 | 0 | Química, materiales y fármacos; los más cercanos tratan carga eléctrica y tráfico urbano, no viento |
| OpenAlex | C1 | 0 | 0 | 0 | Sin resultados |
| OpenAlex | C2 | 2 | 2 | 0 | Efecto climático de parques eólicos y distribución de un insecto marino; ninguno usa regresión por pares |
| OpenAlex | C3 | 14 | 14 | 0 | Lugares llamados *Padre* (Padre Island, Puerto Padre, Misión La Purísima) y estudios de dunas y calidad del aire |
| OpenAlex | C4 | 0 | 0 | 0 | Sin resultados en español |
| OpenAlex | C5 | 1 | 1 | 0 | Regresión de contaminación por partículas con clasificación sinóptica; no usa pares |
| OpenAlex | C6 | 1 | 1 | 0 | Intervalos simultáneos para diferencias de medias, sin relación con pronóstico |
| OpenAlex | C7 | 1 | 1 | 0 | Pronóstico de generación renovable en una microrred con aprendizaje automático; no usa pares |
| arXiv | C1–C5, C7 | 0 | 0 | 0 | Sin resultados |
| arXiv | C6 | 2 | 2 | 0 | Modelos de campos aleatorios espacio-temporales; no usan regresión por pares |
| Semantic Scholar | C1 | 0 | 0 | 0 | Sin resultados |
| Semantic Scholar | C2 | 2 | 2 | 0 | El mismo trabajo sobre parques eólicos y clima que en OpenAlex (dos versiones) |
| Semantic Scholar | C3 | 15 | 15 | 0 | Lugares llamados *Padre* y estudios de dunas, polución y clima |
| Semantic Scholar | C4 | 0 | 0 | 0 | Sin resultados en español |
| Semantic Scholar | C5 | 1 | 1 | 0 | El mismo trabajo de contaminación que en OpenAlex |
| Semantic Scholar | C6 | 1 | 1 | 0 | El mismo trabajo de intervalos simultáneos que en OpenAlex |
| Semantic Scholar | C7 | 10 | 10 | 0 | Trabajos en español de pronóstico de viento con redes neuronales, *wavelets* y posproceso estadístico; ninguno usa diferencias de pares |

En total revisamos `185` registros (con duplicados entre bases) y ninguno aplica la regresión por diferencias de
pares, con ese u otro nombre, al pronóstico de viento.

## 7.3 Declaración de novedad

Al mejor de nuestro conocimiento y según la búsqueda descrita (cinco bases, ocho cadenas en inglés y español,
consultas del 19 y del 22 de octubre de 2026), **no encontramos estudios que apliquen la regresión por diferencias
de pares (PADRE) al pronóstico de la velocidad del viento**, ni a ningún otro problema de pronóstico meteorológico.
Los trabajos más cercanos, que comentamos en la sección siguiente, son el propio PADRE {cite}`tynes2021` (química
computacional, pares sin agrupar), la regresión con redes gemelas {cite}`wetzel2022` (la misma idea con redes
neuronales), el *analog ensemble* {cite}`delle2013` (usa observaciones pasadas del mismo sitio, elegidas por
parecido y sin corrección aprendida) y los estimadores de efectos fijos por diferencias {cite}`wooldridge2010`
(cancelan el efecto de la unidad en modelos lineales). Ninguno forma los pares y las anclas dentro de cada
estación ni usa el resultado como componente de un ensamble de pronóstico.

## 7.4 Trabajos y métodos más cercanos

La búsqueda por nombre no basta para afirmar novedad: un método puede existir en otra área con otro nombre. Por
eso comparamos el modelo con los métodos más cercanos por familia algorítmica, no solo por aplicación.

| Método | Familia | Qué hace | Qué conservamos | Qué es nuevo en nuestro modelo | Por qué no resuelve nuestro problema |
|---|---|---|---|---|---|
| PADRE {cite}`tynes2021` | Meta-algoritmo de regresión por pares | Aprende `y_i − y_j` a partir de `(x_i, x_i − x_j)` y promedia sobre anclas sorteadas en todo el entrenamiento | La forma del regresor de diferencias y el promedio sobre anclas (ecuaciones {eq}`eq-padre-ajuste` y {eq}`eq-padre-prediccion`) | Pares y anclas restringidos a la misma estación, regla de la vecina geográfica para estaciones con poca historia, aplicación a un panel de series de tiempo y uso como componente de un ensamble | Con pares entre estaciones distintas, la diferencia `y_i − y_j` mezcla el cambio de condiciones con el efecto del sitio, que es justo lo que queremos cancelar; además se diseñó para tablas pequeñas sin estructura temporal |
| Regresión con redes gemelas {cite}`wetzel2022` | Redes neuronales por pares | La misma idea de PADRE con una red siamesa que predice diferencias | La idea de predecir diferencias | Motor no neuronal y pares por estación | Usa redes neuronales, que quedan fuera del alcance del proyecto, y no considera grupos ni tiempo |
| *Analog ensemble* {cite}`delle2013` | Métodos de análogos en meteorología | Para un pronóstico numérico nuevo busca días pasados con pronósticos parecidos en el mismo sitio y usa sus observaciones como miembros del ensamble | Usar observaciones pasadas del mismo sitio como referencia | Las anclas no se eligen por parecido: se sortean y una función aprendida `g` corrige la diferencia de condiciones entre la referencia y el caso nuevo | Depende de un pronóstico numérico (no disponible en nuestros datos) y no corrige la diferencia entre el análogo y el caso actual |
| Regresión KNN | Vecindarios | Promedia el objetivo de los `k` vecinos más parecidos | El promedio de objetivos conocidos | Si `g ≡ 0`, PADRE se reduce al promedio de las anclas; el término `g(x, x − x_j)` corrige cada ancla por cuánto difieren sus condiciones | Con vecinos de otras estaciones el promedio arrastra el nivel de otros sitios; en el capítulo 5 KNN fue de los modelos más débiles |
| Random Forest de efectos mixtos (MERF) {cite}`hajjem2014` | Árboles con efectos aleatorios | Estima un efecto aleatorio por grupo con un algoritmo EM alrededor de un Random Forest | Tratar la estación como grupo con un efecto propio | El efecto de la estación no se estima: se cancela al diferenciar y vuelve a entrar por las anclas | Estimar el efecto exige historia suficiente por grupo; lo probamos (sección 6.8) y perdió con Random Forest por una estación con `11` filas |
| Estimadores de efectos fijos y primeras diferencias {cite}`wooldridge2010` | Econometría de panel | Eliminan el efecto fijo de cada unidad restando la media de la unidad o la observación anterior | La cancelación del efecto de la unidad al diferenciar dentro de ella | Las diferencias son entre pares sorteados (no consecutivos), el modelo de la diferencia es no paramétrico y el nivel se recupera con las anclas para poder pronosticar | Son modelos lineales para estimar parámetros, no para pronosticar el nivel de una observación nueva |
| *Stacking* {cite}`wolpert1992` | Combinación de modelos | Aprende pesos o un meta-modelo sobre los pronósticos de los componentes | Combinar varios pronósticos | La combinación es un promedio simple sin pesos; la contribución está en el componente que aporta errores distintos | Con componentes que se equivocan casi igual (correlación de errores de `0.98`), ninguna regla de pesos gana mucho; además los pesos ajustados pueden sobreajustar |
| Aprendizaje con correlación negativa {cite}`liu1999` | Ensambles con diversidad explícita | Entrena los componentes penalizando la correlación de sus errores | Buscar diversidad entre componentes (descomposición {eq}`eq-ambiguedad`) | La diversidad no se impone con una penalización: sale de cambiar lo que el componente aprende (diferencias dentro de la estación) | Se formuló para redes neuronales entrenadas en conjunto; nuestros componentes ya estaban entrenados |
| Rotation Forest {cite}`rodriguez2006` | Ensambles de árboles con rotaciones | Rota grupos de predictoras con PCA antes de cada árbol para diversificar | — | — | Lo probamos (sección 6.8): empata con Random Forest; las rotaciones no cambian la información disponible |

En resumen, la cancelación del efecto de la unidad existe en econometría de panel, el uso de observaciones
pasadas del mismo sitio existe en los métodos de análogos y la regresión por diferencias existe en química. Lo
nuevo es combinarlos en un único algoritmo de pronóstico: aprender con un motor no paramétrico la diferencia
entre horas **de la misma estación**, recuperar el nivel con anclas de esa estación (o de su vecina) y usarlo como
componente que se equivoca distinto dentro de un ensamble.
