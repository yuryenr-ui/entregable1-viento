"""Optimizadores de hiperparámetros con un mismo presupuesto de evaluaciones.

Todos reciben un espacio de ``modelos.ESPACIOS`` y una función objetivo ``objetivo(params) -> RMSE`` que ya
encapsula el preprocesamiento dentro de cada pliegue interno, y devuelven el historial completo de
evaluaciones (configuración, RMSE y tiempo acumulado) para comparar calidad, costo y convergencia.
"""
import itertools
import math
import random
import time

import numpy as np
import optuna
from deap import base, creator, tools

from . import SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)


def decodificar(espacio, u):
    """Convierte un vector ``u`` en ``[0, 1]^d`` en una configuración del espacio.

    Parameters
    ----------
    espacio : dict
        Hiperparámetro -> (tipo, mínimo, máximo) o ("cat", opciones).
    u : sequence of float
        Una coordenada por hiperparámetro, en el orden del espacio.

    Returns
    -------
    dict
        Configuración con los tipos de Python que esperan los estimadores.
    """
    out = {}
    for (k, d), x in zip(espacio.items(), u):
        x = min(max(float(x), 0.0), 1.0)
        if d[0] == "cat":
            out[k] = d[1][min(int(x * len(d[1])), len(d[1]) - 1)]
            continue
        lo, hi = d[1], d[2]
        v = math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))) if d[0].endswith("_log") else lo + x * (hi - lo)
        out[k] = int(round(v)) if d[0].startswith("int") else float(v)
    return out


class _Registro:
    """Guarda cada evaluación con su tiempo y evita repetir configuraciones idénticas."""

    def __init__(self, objetivo):
        self.objetivo, self.historial, self.cache, self.t0 = objetivo, [], {}, time.perf_counter()

    def __call__(self, params, **extra):
        clave = tuple(sorted(params.items()))
        if clave in self.cache:
            return self.cache[clave], False
        t = time.perf_counter()
        rmse = float(self.objetivo(params))
        self.cache[clave] = rmse
        self.historial.append({"evaluacion": len(self.historial) + 1, "rmse": rmse, "params": params,
                               "segundos": time.perf_counter() - t,
                               "segundos_acumulados": time.perf_counter() - self.t0, **extra})
        return rmse, True

    def resultado(self, **extra):
        mejor = min(self.historial, key=lambda h: h["rmse"])
        return {"mejor_params": mejor["params"], "mejor_rmse_interno": mejor["rmse"],
                "evaluaciones": len(self.historial), "historial": self.historial, **extra}


def rejilla(espacio, objetivo, presupuesto, semilla=SEED):
    """Búsqueda en rejilla con a lo sumo ``presupuesto`` puntos.

    Los niveles se reparten de forma voraz en el orden del espacio (importancia esperada): se agrega un nivel
    al siguiente hiperparámetro mientras el producto no supere el presupuesto. Los demás quedan fijos en el
    centro de su rango. Así se hace visible la limitación de la rejilla: con muchos hiperparámetros solo puede
    explorar unos pocos.
    """
    claves = list(espacio)
    niveles = [1] * len(claves)
    mejoro = True
    while mejoro:
        mejoro = False
        for i, k in enumerate(claves):
            tope = len(espacio[k][1]) if espacio[k][0] == "cat" else presupuesto
            prop = niveles.copy(); prop[i] += 1
            if prop[i] <= tope and math.prod(prop) <= presupuesto:
                niveles, mejoro = prop, True
    ejes = []
    for k, L in zip(claves, niveles):
        if espacio[k][0] == "cat":
            n = len(espacio[k][1])
            ejes.append([(j + 0.5) / n for j in range(L)])
        else:
            ejes.append([0.5] if L == 1 else list(np.linspace(0, 1, L)))
    reg = _Registro(objetivo)
    for u in itertools.product(*ejes):
        reg(decodificar(espacio, u))
    return reg.resultado(niveles=dict(zip(claves, niveles)))


def aleatoria(espacio, objetivo, presupuesto, semilla=SEED):
    """Búsqueda aleatoria: ``presupuesto`` configuraciones uniformes (log-uniformes en los tipos ``_log``)."""
    rng = np.random.default_rng(semilla)
    reg = _Registro(objetivo)
    intentos = 0
    while len(reg.historial) < presupuesto and intentos < 50 * presupuesto:
        reg(decodificar(espacio, rng.random(len(espacio))))
        intentos += 1
    return reg.resultado()


def bayesiana(espacio, objetivo, presupuesto, semilla=SEED, n_iniciales=10):
    """Optimización bayesiana con Optuna (TPE multivariado).

    El modelo sustituto es un Tree-structured Parzen Estimator: separa las evaluaciones en buenas (cuantil
    inferior del RMSE) y malas, ajusta una densidad a cada grupo, l(x) y g(x), y propone el candidato que
    maximiza l(x)/g(x), que es equivalente a maximizar la mejora esperada (Expected Improvement). Las primeras
    ``n_iniciales`` evaluaciones son aleatorias para explorar; después el criterio privilegia la explotación
    cerca de las mejores regiones, con exploración por el muestreo de 24 candidatos desde l(x).
    """
    reg = _Registro(objetivo)

    def f(trial):
        params = {}
        for k, d in espacio.items():
            if d[0] == "cat":
                params[k] = trial.suggest_categorical(k, d[1])
            elif d[0].startswith("int"):
                params[k] = trial.suggest_int(k, d[1], d[2], log=d[0].endswith("_log"))
            else:
                params[k] = trial.suggest_float(k, d[1], d[2], log=d[0].endswith("_log"))
        return reg(params)[0]

    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(
        seed=semilla, multivariate=True, n_startup_trials=n_iniciales))
    while len(reg.historial) < presupuesto and len(estudio.trials) < 5 * presupuesto:
        estudio.optimize(f, n_trials=1)
    return reg.resultado(sustituto="TPE", adquisicion="Expected Improvement (l(x)/g(x))")


if not hasattr(creator, "AjusteMin"):
    creator.create("AjusteMin", base.Fitness, weights=(-1.0,))
    creator.create("Individuo", list, fitness=creator.AjusteMin)


def genetica(espacio, objetivo, presupuesto, semilla=SEED, mu=8, lam=8, p_cruce=0.7, p_mut=0.3, torneo=3):
    """Algoritmo genético (μ + λ) con DEAP sobre genes en ``[0, 1]^d``.

    - Selección por torneo de tamaño ``torneo``: presión selectiva moderada y robusta a la escala del RMSE.
    - Cruce SBX acotado (η = 15) con probabilidad ``p_cruce``: hijos cerca de los padres, respetando los límites.
    - Mutación polinomial acotada (η = 20, probabilidad por gen 1/d) con probabilidad ``p_mut``.
    - Elitismo implícito del esquema (μ + λ): padres e hijos compiten y sobreviven los ``mu`` mejores.
    El presupuesto cuenta solo evaluaciones nuevas (las configuraciones repetidas salen del caché). Se registra
    la diversidad genética (desviación estándar media de los genes) en cada generación para detectar una
    convergencia prematura.
    """
    rnd = random.Random(semilla)
    d = len(espacio)
    tb = base.Toolbox()
    tb.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=15.0)
    tb.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0 / d)
    tb.register("select", tools.selTournament, tournsize=torneo)
    reg = _Registro(objetivo)
    random.seed(semilla)  # los operadores de DEAP usan el módulo random global

    def evaluar(ind, gen):
        if not ind.fitness.valid:
            ind.fitness.values = (reg(decodificar(espacio, ind), generacion=gen)[0],)

    pob = [creator.Individuo([rnd.random() for _ in range(d)]) for _ in range(mu)]
    for ind in pob:
        evaluar(ind, 0)
    diversidad = [{"generacion": 0, "diversidad": float(np.mean(np.std(np.array(pob), axis=0))),
                   "mejor": min(i.fitness.values[0] for i in pob), "evaluaciones": len(reg.historial)}]
    gen = 0
    while len(reg.historial) < presupuesto and gen < 50:
        gen += 1
        hijos = []
        while len(hijos) < lam:
            a, b = [tb.clone(x) for x in tb.select(pob, 2)]
            if rnd.random() < p_cruce:
                tb.mate(a, b)
                del a.fitness.values, b.fitness.values
            for h in (a, b):
                if rnd.random() < p_mut:
                    tb.mutate(h)
                    del h.fitness.values
            hijos.extend([a, b])
        for h in hijos[:lam]:
            if len(reg.historial) >= presupuesto:
                break
            evaluar(h, gen)
        validos = [h for h in hijos[:lam] if h.fitness.valid]
        pob = tools.selBest(pob + validos, mu)
        diversidad.append({"generacion": gen, "diversidad": float(np.mean(np.std(np.array(pob), axis=0))),
                           "mejor": pob[0].fitness.values[0], "evaluaciones": len(reg.historial)})
    return reg.resultado(diversidad=diversidad, operadores={
        "seleccion": f"torneo (tamaño {torneo})", "cruce": f"SBX acotado, η = 15, p = {p_cruce}",
        "mutacion": f"polinomial acotada, η = 20, p = {p_mut}, 1/d por gen", "elitismo": "(μ + λ)",
        "mu": mu, "lambda": lam})


def eliminacion_sucesiva(espacio, objetivo_fidelidad, n_inicial=27, eta=3, semilla=SEED):
    """Successive Halving: muchas configuraciones con presupuesto parcial y solo las mejores llegan al total.

    ``objetivo_fidelidad(params, fraccion)`` evalúa con una fracción de las filas de ajuste. Con ``eta = 3`` y
    27 configuraciones iniciales hay tres rondas (1/9, 1/3 y todas las filas), quedando un tercio en cada una.
    El costo equivale a ``27/9 + 9/3 + 3 = 9`` evaluaciones completas. La métrica de fidelidad es el RMSE
    interno, la misma del objetivo final.
    """
    rng = np.random.default_rng(semilla)
    rondas = int(round(math.log(n_inicial, eta)))
    candidatos = [decodificar(espacio, rng.random(len(espacio))) for _ in range(n_inicial)]
    historial, t0 = [], time.perf_counter()
    for r in range(rondas):
        fr = eta ** (r - rondas + 1)
        puntajes = []
        for c in candidatos:
            t = time.perf_counter()
            s = float(objetivo_fidelidad(c, fr))
            puntajes.append(s)
            historial.append({"ronda": r, "fraccion": fr, "rmse": s, "params": c,
                              "segundos": time.perf_counter() - t,
                              "segundos_acumulados": time.perf_counter() - t0})
        orden = np.argsort(puntajes)
        candidatos = [candidatos[i] for i in orden[:max(1, len(candidatos) // eta)]]
    finales = [h for h in historial if h["fraccion"] == 1]
    mejor = min(finales, key=lambda h: h["rmse"])
    return {"mejor_params": mejor["params"], "mejor_rmse_interno": mejor["rmse"], "historial": historial,
            "evaluaciones_equivalentes": sum(h["fraccion"] for h in historial)}


OPTIMIZADORES = {"Rejilla": rejilla, "Aleatoria": aleatoria, "Bayesiana (Optuna)": bayesiana,
                 "Genética (DEAP)": genetica}
