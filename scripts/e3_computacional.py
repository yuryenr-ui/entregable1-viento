"""Experimentos de optimización computacional de la sección 5.3.

Uso (desde la carpeta entregable1):
    python scripts/e3_computacional.py

Guarda en resultados/e3/computacional/ las tablas de tiempos: escalamiento con n, variantes estándar frente a
optimizadas de cada modelo, paralelismo, GPU frente a CPU y perfilamiento de una evaluación. Cada tiempo es la
mediana de varias repeticiones para reducir la variabilidad de la medición.
"""
import cProfile
import io
import pstats
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed, parallel_backend
import threading

import psutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import SEED                                     # noqa: E402
from e3.datos import RESULTADOS, cargar, corte, TRIMESTRES  # noqa: E402

if __name__ == "__main__":  # en Windows los procesos hijos vuelven a importar el script
    warnings.filterwarnings("ignore")
    SALIDA = RESULTADOS / "computacional"
    SALIDA.mkdir(parents=True, exist_ok=True)
    TR, TE, F = cargar()
    X, y = TR[F].to_numpy(np.float32), TR.y.to_numpy()
    Xv, yv = TE[F].to_numpy(np.float32)[:10000], TE.y.to_numpy()[:10000]
    rng = np.random.default_rng(SEED)
    orden = rng.permutation(len(X))


    def cronometrar(fabrica, n, repeticiones=3, n_pred=10000):
        """Mediana de los tiempos de ajuste con n filas y de predicción de ``n_pred`` filas, y el RMSE."""
        idx = orden[:n]
        tf, tp = [], []
        for _ in range(repeticiones):
            m = fabrica()
            t = time.perf_counter(); m.fit(X[idx], y[idx]); tf.append(time.perf_counter() - t)
            t = time.perf_counter(); p = m.predict(Xv[:n_pred]); tp.append(time.perf_counter() - t)
        return {"n": n, "ajuste_s": float(np.median(tf)), "prediccion_s": float(np.median(tp)),
                "rmse": float(np.sqrt(np.mean((yv[:n_pred] - p) ** 2)))}


    def memoria_pico(fabrica, n):
        """Incremento máximo de memoria (MB) durante el ajuste con n filas."""
        # se muestrea la memoria residente del propio proceso desde un hilo (memory_profiler lanza un proceso
        # nuevo que en Windows vuelve a importar el script completo)
        idx = orden[:n]
        proc, muestras, fin = psutil.Process(), [], threading.Event()

        def muestrear():
            while not fin.is_set():
                muestras.append(proc.memory_info().rss)
                fin.wait(0.05)
        base = proc.memory_info().rss
        h = threading.Thread(target=muestrear)
        h.start()
        fabrica().fit(X[idx], y[idx])
        fin.set()
        h.join()
        return float((max(muestras + [base]) - base) / 2**20)


    # configuraciones representativas (centro de los espacios de búsqueda)
    FABRICAS = {
        "KNN": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(50, algorithm="kd_tree", n_jobs=4)),
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(1.0)),
        "Lasso": lambda: make_pipeline(StandardScaler(), Lasso(1e-3)),
        "Árbol de decisión": lambda: DecisionTreeRegressor(max_depth=15, min_samples_leaf=50, random_state=SEED),
        "Random Forest": lambda: RandomForestRegressor(150, max_depth=15, min_samples_leaf=50, max_features=0.5,
                                                       n_jobs=4, random_state=SEED),
        "XGBoost": lambda: xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, tree_method="hist",
                                            device="cuda", random_state=SEED),
        "SVR": lambda: make_pipeline(StandardScaler(), LinearSVR(C=0.01, epsilon=0, dual=False,
                                                                 loss="squared_epsilon_insensitive")),
    }

    # 1. escalamiento del tiempo de ajuste con n (exponente empírico = pendiente en escala log-log)
    print("1. escalamiento", flush=True)
    if not (SALIDA / "escalamiento.csv").exists():  # se salta si ya se calculó en una corrida anterior
        filas = []
        for nombre, f in FABRICAS.items():
            for n in [10000, 20000, 40000, 80000, 160000, 320000]:
                filas.append({"modelo": nombre, **cronometrar(f, n, repeticiones=2 if nombre == "Random Forest" else 3)})
                print(" ", filas[-1], flush=True)
        pd.DataFrame(filas).to_csv(SALIDA / "escalamiento.csv", index=False)

    # 2. memoria pico durante el ajuste con 160 mil filas
    print("2. memoria", flush=True)
    pd.DataFrame([{"modelo": k, "memoria_MB": memoria_pico(f, 160000)} for k, f in FABRICAS.items()]).to_csv(
        SALIDA / "memoria.csv", index=False)

    # 3. variantes estándar frente a optimizadas
    print("3. variantes", flush=True)
    var = []


    def variante(modelo, nombre, fabrica, n, **kw):
        r = cronometrar(fabrica, n, **kw)
        var.append({"modelo": modelo, "variante": nombre, **r})
        print(" ", var[-1], flush=True)


    # KNN: búsqueda exhaustiva frente a estructuras de indexación (resultado exacto en los tres casos)
    for alg in ["brute", "kd_tree", "ball_tree"]:
        variante("KNN", alg, lambda a=alg: make_pipeline(StandardScaler(), KNeighborsRegressor(50, algorithm=a, n_jobs=4)),
                 160000, repeticiones=2)
    # Ridge: ecuaciones normales (Cholesky) frente a SAGA y gradiente conjugado
    for solver in ["cholesky", "saga", "sparse_cg"]:
        variante("Ridge", solver, lambda s=solver: make_pipeline(StandardScaler(), Ridge(1.0, solver=s, random_state=SEED)),
                 320000)
    # Lasso: descenso por coordenadas frente a SGD con penalización L1
    variante("Lasso", "descenso por coordenadas", FABRICAS["Lasso"], 320000)
    variante("Lasso", "SGD (L1)", lambda: make_pipeline(StandardScaler(), SGDRegressor(penalty="l1", alpha=1e-5,
                                                                                        random_state=SEED)), 320000)
    # XGBoost: método exacto frente a histogramas, en CPU y en GPU
    for nombre, kw in [("exact (CPU)", dict(tree_method="exact")), ("hist (CPU)", dict(tree_method="hist", device="cpu")),
                       ("hist (GPU)", dict(tree_method="hist", device="cuda"))]:
        variante("XGBoost", nombre, lambda k=kw: xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                                                  random_state=SEED, n_jobs=16, **k), 160000, repeticiones=2)
    # SVR: núcleo RBF completo (O(n²)–O(n³)) frente a Nyström + primal y al SVR lineal (primal o SGD)
    for n in [2500, 5000, 10000, 20000]:
        variante("SVR", f"núcleo RBF completo (n = {n})", lambda: make_pipeline(StandardScaler(), SVR(C=1.0, gamma=0.02)),
                 n, repeticiones=1, n_pred=2000)
        variante("SVR", f"Nyström 300 + primal (n = {n})", lambda: make_pipeline(
            StandardScaler(), Nystroem(gamma=0.02, n_components=300, random_state=SEED),
            LinearSVR(C=1.0, epsilon=0, dual=False, loss="squared_epsilon_insensitive")), n, repeticiones=1, n_pred=2000)
    variante("SVR", "lineal primal (LinearSVR)", FABRICAS["SVR"], 320000)
    variante("SVR", "lineal SGD", lambda: make_pipeline(StandardScaler(), SGDRegressor(
        loss="squared_epsilon_insensitive", epsilon=0, alpha=1e-4, random_state=SEED)), 320000)
    pd.DataFrame(var).to_csv(SALIDA / "variantes.csv", index=False)

    # 4. XGBoost con parada temprana (validación independiente: 2024-T4) frente a un número fijo de árboles
    print("4. parada temprana", flush=True)
    tr, va = corte(TR, TRIMESTRES[7], TRIMESTRES[8])
    tr_e, va_e = tr[TR.t_y.to_numpy()[tr] < TRIMESTRES[6]], tr[TR.t.to_numpy()[tr] >= TRIMESTRES[6]]  # validación: 2024-T3
    filas = []
    for nombre, kw, fit_kw in [("1500 árboles fijos", {}, {}),
                               ("parada temprana (50 rondas)", {"early_stopping_rounds": 50},
                                {"eval_set": [(X[va_e], y[va_e])], "verbose": False})]:
        m = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=6, tree_method="hist", device="cuda",
                             random_state=SEED, **kw)
        t = time.perf_counter(); m.fit(X[tr_e], y[tr_e], **fit_kw); s = time.perf_counter() - t
        p = m.predict(X[va])
        filas.append({"variante": nombre, "ajuste_s": s, "arboles": int(getattr(m, "best_iteration", 1499) or 1499) + 1,
                      "rmse_2024T4": float(np.sqrt(np.mean((y[va] - p) ** 2)))})
    pd.DataFrame(filas).to_csv(SALIDA / "parada_temprana.csv", index=False)

    # 5. paralelismo: Random Forest con distintos núcleos y 3 ajustes de árbol en serie o en paralelo
    print("5. paralelismo", flush=True)
    filas = []
    for nj in [1, 2, 4, 8, 16]:
        r = cronometrar(lambda n=nj: RandomForestRegressor(100, max_depth=15, min_samples_leaf=50, max_features=0.5,
                                                           n_jobs=n, random_state=SEED), 160000, repeticiones=2)
        filas.append({"experimento": "Random Forest n_jobs", "configuracion": nj, "segundos": r["ajuste_s"]})
    idx3 = [orden[i * 60000:(i + 1) * 60000] for i in range(6)]


    def ajustar_arbol(i):
        return DecisionTreeRegressor(max_depth=15, min_samples_leaf=50, random_state=SEED).fit(X[idx3[i]], y[idx3[i]])


    t = time.perf_counter(); [ajustar_arbol(i) for i in range(6)]
    filas.append({"experimento": "6 ajustes de árbol", "configuracion": "en serie", "segundos": time.perf_counter() - t})
    for backend in ["loky", "threading"]:
        with parallel_backend(backend, n_jobs=6):
            t = time.perf_counter(); Parallel()(delayed(ajustar_arbol)(i) for i in range(6))
        filas.append({"experimento": "6 ajustes de árbol", "configuracion": f"joblib {backend}",
                      "segundos": time.perf_counter() - t})
    pd.DataFrame(filas).to_csv(SALIDA / "paralelismo.csv", index=False)

    # 6. perfilamiento de una evaluación del objetivo (3 pliegues internos) para KNN y Random Forest
    print("6. perfilamiento", flush=True)
    salida_txt = []
    for nombre in ["KNN", "Random Forest"]:
        pr = cProfile.Profile()
        pr.enable()
        for i in range(3):
            m = FABRICAS[nombre]().fit(X[idx3[i][:40000]], y[idx3[i][:40000]])
            m.predict(X[idx3[i + 3][:10000]])
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
        salida_txt.append(f"===== {nombre} =====\n{s.getvalue()}")
    (SALIDA / "perfil.txt").write_text("\n".join(salida_txt), encoding="utf-8")
    print("listo", flush=True)
