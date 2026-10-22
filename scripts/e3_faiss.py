"""KNN con FAISS: búsqueda exacta y aproximada de vecinos frente a scikit-learn.

Uso (desde la carpeta entregable1, con la máquina sin otros procesos pesados):
    python scripts/e3_faiss.py

Mismo protocolo que la sección 3 de ``e3_computacional.py``: 160 mil filas de ajuste, 10 mil consultas de 2025,
k = 50 y escalado estándar, mediana de 2 repeticiones. Además del tiempo y el RMSE se mide el *recall*: la
fracción de los 50 vecinos exactos que encuentra cada método, que cuantifica el costo de la aproximación.
Resultado en resultados/e3/computacional/knn_faiss.csv.
"""
import sys
import time
import warnings
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import SEED                       # noqa: E402
from e3.datos import RESULTADOS, cargar   # noqa: E402

warnings.filterwarnings("ignore")
K, N, N_PRED, REPETICIONES = 50, 160000, 10000, 2


class BuscadorFaiss:
    """Índice FAISS que devuelve los ``k`` vecinos más cercanos (distancia euclidiana)."""

    def __init__(self, tipo):
        self.tipo = tipo

    def fit(self, Z):
        d = Z.shape[1]
        if self.tipo == "exacto (IndexFlatL2)":
            self.indice = faiss.IndexFlatL2(d)
        elif self.tipo == "aproximado (HNSW, M = 32)":
            self.indice = faiss.IndexHNSWFlat(d, 32)
            self.indice.hnsw.efSearch = 128
        else:                                  # IVF: 400 celdas, se revisan las 8 más cercanas a cada consulta
            self.indice = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 400)
            self.indice.train(Z)
            self.indice.nprobe = 8
        self.indice.add(Z)
        return self

    def vecinos(self, Q, k):
        return self.indice.search(Q, k)[1]


class BuscadorSklearn:
    def __init__(self, alg):
        self.alg = alg

    def fit(self, Z):
        self.nn = NearestNeighbors(n_neighbors=K, algorithm=self.alg, n_jobs=4).fit(Z)
        return self

    def vecinos(self, Q, k):
        return self.nn.kneighbors(Q, k, return_distance=False)


if __name__ == "__main__":
    faiss.omp_set_num_threads(4)              # mismos 4 hilos que KNeighborsRegressor(n_jobs=4)
    TR, TE, F = cargar()
    orden = np.random.default_rng(SEED).permutation(len(TR))[:N]
    esc = StandardScaler().fit(TR[F].to_numpy(np.float32)[orden])
    Z = esc.transform(TR[F].to_numpy(np.float32)[orden]).astype(np.float32)
    y = TR.y.to_numpy()[orden]
    Q = esc.transform(TE[F].to_numpy(np.float32)[:N_PRED]).astype(np.float32)
    yq = TE.y.to_numpy()[:N_PRED]
    exactos = None
    filas = []
    for nombre, b in [("scikit-learn brute", BuscadorSklearn("brute")),
                      ("FAISS exacto (IndexFlatL2)", BuscadorFaiss("exacto (IndexFlatL2)")),
                      ("FAISS aproximado (IVF, 400 celdas, nprobe = 8)", BuscadorFaiss("ivf")),
                      ("FAISS aproximado (HNSW, M = 32)", BuscadorFaiss("aproximado (HNSW, M = 32)"))]:
        ta, tp = [], []
        for _ in range(REPETICIONES):
            t = time.perf_counter(); b.fit(Z); ta.append(time.perf_counter() - t)
            t = time.perf_counter(); idx = b.vecinos(Q, K); tp.append(time.perf_counter() - t)
        if exactos is None:
            exactos = idx
        recall = np.mean([len(set(a) & set(e)) / K for a, e in zip(idx, exactos)])
        p = y[idx].mean(1)
        filas.append({"variante": nombre, "ajuste_s": float(np.median(ta)), "prediccion_s": float(np.median(tp)),
                      "recall@50": float(recall), "rmse": float(np.sqrt(np.mean((yq - p) ** 2)))})
        print(filas[-1], flush=True)
    pd.DataFrame(filas).to_csv(RESULTADOS / "computacional" / "knn_faiss.csv", index=False)
