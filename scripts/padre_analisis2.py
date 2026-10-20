"""Segunda parte de los análisis del modelo propuesto: lo que la guía del Entregable 3 pide y aún faltaba.

Uso (desde la carpeta entregable1, después de ``padre_analisis.py``):
    python scripts/padre_analisis2.py

1. IC 95 % del RMSE de 2025 con bootstrap estacionario por días (los 7 modelos y el propuesto) y de la
   diferencia de RMSE frente a XGBoost y Random Forest.
2. Reality Check y SPA con tres referencias: XGBoost (elegido en el Entregable 3), Random Forest (mejor del
   curso en 2025) y el propuesto (¿algún modelo del curso lo supera?).
3. Figuras: entrenamiento, validación y prueba; residuos; diagrama de diferencia crítica; convergencia.
4. SHAP (explicador por permutación sobre la mezcla completa) y LIME para el modelo propuesto.
Tablas y figuras en resultados/e3/padre_analisis/.
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scikit_posthocs as sp  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import SEED  # noqa: E402
from e3 import evaluacion as ev  # noqa: E402
from e3.datos import RESULTADOS, TRIMESTRES, cargar, pliegues_externos, submuestra  # noqa: E402
from e3.modelos import ESPACIOS, construir  # noqa: E402
from e3.padre import PadreEstacion  # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
SALIDA = RESULTADOS / "padre_analisis"
FIG = SALIDA / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
ANIDADO = RESULTADOS / "padre_anidado"
PROPUESTO = "PADRE por estación + XGBoost"
PLIEGUES = ["2024-T1", "2024-T2", "2024-T3", "2024-T4"]
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

TR, TE, F = cargar()
X, y, est = TR[F].to_numpy(np.float32), TR.y.to_numpy(), TR.est.to_numpy()
X_te, y_te, est_te = TE[F].to_numpy(np.float32), TE.y.to_numpy(), TE.est.to_numpy()
co = pd.read_csv(RESULTADOS / "estaciones.csv", dtype={"est": str}).set_index("est")
fin = json.loads((ANIDADO / "PADRE_XGB__final.json").read_text(encoding="utf-8"))
W, MEJOR = fin["peso_padre"], fin["mejor_params"]
tm = pd.read_csv(RESULTADOS / "tabla_maestra_resumen.csv")
OPT = dict(tm[tm.elegido].set_index("modelo").optimizador)


def archivo(modelo, optimizador, pliegue):
    nombre = f"{modelo}__{optimizador}__{pliegue}".replace(" ", "_").replace("(", "").replace(")", "")
    return RESULTADOS / "corridas" / nombre


def guardar(df, nombre):
    df.to_csv(SALIDA / f"{nombre}.csv")
    print(f"\n### {nombre}\n{df.round(4).to_string()}", flush=True)


def figura(nombre):
    plt.tight_layout()
    plt.savefig(FIG / f"{nombre}.png", bbox_inches="tight")
    plt.close()
    print("figura", nombre, flush=True)


d_fin = np.load(ANIDADO / "PADRE_XGB__final.npz")
pred = {PROPUESTO: d_fin["mezcla"].astype(float)}
pred |= {m: np.load(f"{archivo(m, o, 'final')}.npy").astype(float) for m, o in OPT.items()}
L = ev.perdida_diaria(TE.t, y_te, pred)

# ---------------------------------------------------------------- 1. IC del RMSE
from arch.bootstrap import StationaryBootstrap  # noqa: E402

dia = TE.t.dt.floor("D").to_numpy()
sse = pd.DataFrame({k: (y_te - p) ** 2 for k, p in pred.items()}).groupby(dia).sum()
n_dia = pd.Series(1, index=dia).groupby(level=0).sum().reindex(sse.index).to_numpy()
bs = StationaryBootstrap(7, sse.to_numpy(), n_dia, seed=SEED)
reps = np.array([np.sqrt(s.sum(0) / n.sum()) for (s, n), _ in bs.bootstrap(2000)])
cols = list(sse.columns)
filas = []
for j, m in enumerate(cols):
    fila = {"modelo": m, "RMSE": rmse(y_te, pred[m]), "IC 95 % inf": np.percentile(reps[:, j], 2.5),
            "IC 95 % sup": np.percentile(reps[:, j], 97.5)}
    for ref in ["XGBoost", "Random Forest"]:
        d = reps[:, cols.index(ref)] - reps[:, j]
        fila[f"Δ vs {ref} (IC 95 %)"] = f"{np.percentile(d, 2.5):+.4f} – {np.percentile(d, 97.5):+.4f}"
    filas.append(fila)
guardar(pd.DataFrame(filas).set_index("modelo").sort_values("RMSE"), "8_ic_rmse")

# ---------------------------------------------------------------- 2. Reality Check y SPA
todos = [PROPUESTO, *OPT]
spa = {ref: ev.spa(L[ref], L[[m for m in todos if m != ref]]) for ref in ["XGBoost", "Random Forest", PROPUESTO]}
spa = pd.DataFrame(spa).T
spa.index.name = "referencia (los demás 7 modelos como alternativas)"
guardar(spa, "9_spa")

# ---------------------------------------------------------------- 3a. entrenamiento, validación y prueba
modelo = PadreEstacion({k: MEJOR[k] for k in ESPACIOS["XGBoost"]}, co, MEJOR["pares"], MEJOR["anclas"]).fit(X, y, est)
xgb_fin = json.loads(Path(f"{archivo('XGBoost', 'Aleatoria', 'final')}.json").read_text(encoding="utf-8"))
m_xgb = construir("XGBoost", xgb_fin["mejor_params"], n_jobs=8).fit(X, y)
p_tr = W * modelo.predict(X, est) + (1 - W) * m_xgb.predict(X)
print(f"RMSE entrenamiento {rmse(y, p_tr):.4f} | 2025 {rmse(y_te, pred[PROPUESTO]):.4f}", flush=True)
EXT = pliegues_externos(TR)
val_idx = np.concatenate([va for _, _, va in EXT])
p_val = np.concatenate([np.load(ANIDADO / f"PADRE_XGB__{n}.npz")["mezcla"] for n in PLIEGUES])
cuenta = TR.est.iloc[val_idx].value_counts().add(TE.est.value_counts(), fill_value=0)
EST = cuenta.idxmax()
serie = pd.concat([
    pd.DataFrame({"t": TR.t_y, "y": y, "p": p_tr, "tramo": "entrenamiento (ajuste)", "est": TR.est}),
    pd.DataFrame({"t": TR.t_y.iloc[val_idx].to_numpy(), "y": y[val_idx], "p": p_val,
                  "tramo": "validación (bucle externo 2024)", "est": TR.est.iloc[val_idx].to_numpy()}),
    pd.DataFrame({"t": TE.t_y, "y": y_te, "p": pred[PROPUESTO], "tramo": "prueba (2025)", "est": TE.est})])
s = serie[serie.est == EST]
s = s[~((s.tramo == "entrenamiento (ajuste)") & (s.t >= TRIMESTRES[4]))]
fig, axs = plt.subplots(2, 1, figsize=(12, 7))
dd = s.groupby(["tramo", s.t.dt.floor("D")])[["y", "p"]].mean().reset_index()
obs = s.groupby(s.t.dt.floor("D")).y.mean().asfreq("D")
axs[0].plot(obs.index, obs, color="0.6", lw=.7, label="observado")
for tr_, c in zip(["entrenamiento (ajuste)", "validación (bucle externo 2024)", "prueba (2025)"], ["C0", "C2", "C3"]):
    d = dd[dd.tramo == tr_]
    axs[0].plot(d.t, d.p, ".", ms=2.5, color=c, label=f"predicción: {tr_}")
axs[0].axvline(TRIMESTRES[4], color="k", ls="--", lw=.8)
axs[0].axvline(pd.Timestamp("2025-01-01"), color="k", ls="--", lw=.8)
axs[0].set(ylabel="m/s (media diaria)", title=f"{PROPUESTO} en la estación {EST}",
           xlim=(pd.Timestamp("2022-01-01"), None))
axs[0].legend(fontsize=8, ncol=2)
z = s[s.tramo == "prueba (2025)"].set_index("t").sort_index()
z = z[z.index >= z.index[len(z) // 2]].iloc[:24 * 14]
axs[1].plot(z.index, z.y, color="0.4", label="observado")
axs[1].plot(z.index, z.p, color="C3", label="propuesto")
axs[1].plot(z.index, pd.Series(pred["XGBoost"], index=pd.MultiIndex.from_arrays([TE.est, TE.t_y])).loc[EST]
            .reindex(z.index), color="C0", lw=.8, ls="--", label="XGBoost")
axs[1].set(ylabel="m/s", title="Dos semanas de 2025 (horario)")
axs[1].legend(fontsize=8)
figura("a_entrenamiento_validacion_prueba")

# ---------------------------------------------------------------- 3b. residuos
e_te = y_te - pred[PROPUESTO]
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
axs[0].hist(e_te, bins=120, density=True, color="C0", alpha=.7)
xx = np.linspace(-4, 4, 200)
axs[0].plot(xx, stats.norm.pdf(xx, e_te.mean(), e_te.std()), "k--", lw=1, label="normal")
axs[0].set(xlabel="residuo (m/s)", title="Histograma de residuos (2025)", xlim=(-4, 4))
axs[0].legend()
r_est = pd.Series(e_te, index=pd.MultiIndex.from_arrays([TE.est, TE.t])).loc[EST].sort_index()
r_est = r_est.reindex(pd.date_range(r_est.index.min(), r_est.index.max(), freq="h"))
plot_acf(r_est.to_numpy(), lags=72, missing="conservative", ax=axs[1], markersize=3)
axs[1].set(title=f"ACF de residuos, estación {EST}", xlabel="rezago (h)")
q = pd.qcut(pred[PROPUESTO], 20)
g = pd.DataFrame({"p": pred[PROPUESTO], "e": e_te}).groupby(q, observed=True).agg(
    p=("p", "mean"), sd=("e", "std"), m=("e", "mean"))
axs[2].plot(g.p, g.sd, "o-", label="desviación del residuo")
axs[2].plot(g.p, g.m, "s-", label="media del residuo")
axs[2].axhline(0, color="k", lw=.6)
axs[2].set(xlabel="pronóstico (m/s, 20 grupos)", ylabel="m/s", title="Residuo según el pronóstico")
axs[2].legend(fontsize=8)
figura("b_residuos")

# ---------------------------------------------------------------- 3c. diagrama de diferencia crítica (2024)
bloques = []
for n, _, va in EXT:
    d = pd.DataFrame({"est": est[va], "y": y[va]})
    d[PROPUESTO] = (d.y - np.load(ANIDADO / f"PADRE_XGB__{n}.npz")["mezcla"]) ** 2
    for m, o in OPT.items():
        d[m] = (d.y - np.load(f"{archivo(m, o, n)}.npy")) ** 2
    gg = d.groupby("est")
    b = np.sqrt(gg[todos].mean())[gg.size() >= 48]
    b.index = [f"{e} · {n}" for e in b.index]
    bloques.append(b)
fr = ev.friedman_nemenyi(pd.concat(bloques))
fig, ax = plt.subplots(figsize=(9, 2.8))
sp.critical_difference_diagram(fr["rangos"], fr["nemenyi"], ax=ax, label_fmt_left="{label} ({rank:.2f})  ",
                               label_fmt_right="  ({rank:.2f}) {label}")
ax.set_title(f"Diferencia crítica con el modelo propuesto (Nemenyi, α = 0.05, CD = {fr['CD']:.2f})")
figura("c_diferencia_critica")

# ---------------------------------------------------------------- 3d. convergencia de la búsqueda
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for n in PLIEGUES + ["final"]:
    h = pd.DataFrame(json.loads((ANIDADO / f"PADRE_XGB__{n}.json").read_text(encoding="utf-8"))["historial"])
    axs[0].step(h.evaluacion, h.rmse.cummin(), where="post", label=n)
axs[0].set(xlabel="evaluación", ylabel="mejor RMSE interno (con w óptimo)", title="Propuesto: búsqueda por pliegue")
axs[0].legend(fontsize=8)
for nombre, ruta in [(PROPUESTO, ANIDADO / "PADRE_XGB__final.json"),
                     ("XGBoost (aleatoria)", Path(f"{archivo('XGBoost', 'Aleatoria', 'final')}.json"))]:
    h = pd.DataFrame(json.loads(ruta.read_text(encoding="utf-8"))["historial"])
    axs[1].step(h.segundos_acumulados / 60, h.rmse.cummin(), where="post", label=nombre)
axs[1].set(xlabel="minutos de búsqueda", ylabel="mejor RMSE interno", title="Búsqueda final (2024 como bucle interno)")
axs[1].legend(fontsize=8)
figura("d_convergencia")

# ---------------------------------------------------------------- 4. SHAP y LIME
import shap  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402

estacion_actual = [None]


def f_mezcla(Z):
    """Pronóstico del modelo propuesto para filas de una misma estación (la del pronóstico que se explica)."""
    Z = np.asarray(Z, np.float32)
    return W * modelo.predict(Z, np.repeat(estacion_actual[0], len(Z))) + (1 - W) * m_xgb.predict(Z)


idx = submuestra(np.arange(len(TE)), 200, SEED)
fondo = X[submuestra(np.arange(len(TR)), 50, SEED)]
explicador = shap.explainers.Permutation(f_mezcla, shap.maskers.Independent(fondo), feature_names=F, seed=SEED)
valores = []
for k, i in enumerate(idx):
    estacion_actual[0] = est_te[i]
    valores.append(explicador(X_te[i:i + 1], max_evals=20 * len(F) + 1))   # unas 10 permutaciones
    if k % 50 == 0:
        print("SHAP", k, flush=True)
sv = shap.Explanation(np.vstack([v.values for v in valores]), np.concatenate([v.base_values for v in valores]),
                      X_te[idx], feature_names=F)
shap.summary_plot(sv.values, pd.DataFrame(X_te[idx], columns=F), max_display=15, show=False, plot_size=(9, 6))
plt.title(f"SHAP del modelo propuesto en 200 pronósticos de 2025")
figura("e_shap_resumen")
m_xgb.get_booster().set_param({"device": "cpu"})
import xgboost as xgb  # noqa: E402

sv_xgb = m_xgb.get_booster().predict(xgb.DMatrix(X_te[idx]), pred_contribs=True)[:, :-1]
imp = pd.DataFrame({f"media |SHAP| {PROPUESTO}": np.abs(sv.values).mean(0),
                    "media |SHAP| XGBoost": np.abs(sv_xgb).mean(0)}, index=F).sort_values(
    f"media |SHAP| {PROPUESTO}", ascending=False)
guardar(imp, "10_shap_importancia")
err = np.abs(y_te[idx] - pred[PROPUESTO][idx])
for nombre, j in [("bien_predicho", int(np.argmin(err))), ("error_alto", int(np.argmax(err)))]:
    i = idx[j]
    shap.plots.waterfall(sv[j], max_display=10, show=False)
    plt.title(f"{nombre.replace('_', ' ')}: ŷ = {pred[PROPUESTO][i]:.2f}, y = {y_te[i]:.2f} m/s, estación {est_te[i]}")
    figura(f"f_shap_{nombre}")

ref = submuestra(np.arange(len(TR)), 20000, SEED)
lime_exp = LimeTabularExplainer(X[ref], feature_names=F, mode="regression", discretize_continuous=True,
                                random_state=SEED)
pp = pred[PROPUESTO]
casos = {"calma (percentil 10)": np.quantile(pp, .1), "típico (mediana)": np.quantile(pp, .5),
         "viento fuerte (percentil 99)": np.quantile(pp, .99)}
fig, axs = plt.subplots(1, 3, figsize=(16, 3.8))
for ax, (nombre, q_) in zip(axs, casos.items()):
    i = int(np.argmin(np.abs(pp - q_)))
    estacion_actual[0] = est_te[i]
    exp = lime_exp.explain_instance(X_te[i], f_mezcla, num_features=8, num_samples=3000)
    lst = exp.as_list()[::-1]
    ax.barh([a for a, _ in lst], [b for _, b in lst], color=["C3" if b < 0 else "C2" for _, b in lst])
    ax.set(title=f"{nombre}\nŷ = {pp[i]:.2f}, y = {y_te[i]:.2f} m/s, R² local = {exp.score:.2f}",
           xlabel="contribución local (m/s)")
    ax.tick_params(axis="y", labelsize=7)
figura("g_lime")
