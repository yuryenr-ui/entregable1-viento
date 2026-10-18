"""Métricas, diagnóstico de residuos y pruebas estadísticas para comparar pronósticos.

Las pruebas que comparan pérdidas usan la serie diaria de la pérdida media (todas las estaciones-hora de cada
día de 2025): así la unidad de remuestreo es el día, como en el bootstrap por bloques de la sección 3.2.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import SEED


def metricas(y, p, pers=None):
    """RMSE, MAE, R² y, si se da la persistencia, la reducción del RMSE frente a ella."""
    e = y - p
    out = {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
           "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}
    if pers is not None:
        out["Mejora vs. persistencia"] = 1 - out["RMSE"] / float(np.sqrt(np.mean((y - pers) ** 2)))
    return out


def perdida_diaria(t, y, preds):
    """Error cuadrático medio por día para varios pronósticos.

    Parameters
    ----------
    t : array-like de fechas de emisión.
    y : numpy.ndarray, observado.
    preds : dict nombre -> numpy.ndarray de pronósticos.

    Returns
    -------
    pandas.DataFrame
        Una fila por día y una columna por pronóstico.
    """
    dia = pd.to_datetime(pd.Series(t)).dt.floor("D").to_numpy()
    return pd.DataFrame({k: (y - v) ** 2 for k, v in preds.items()}).groupby(dia).mean()


# ------------------------------------------------------------------ residuos

def prueba_white(residuos, X, n=20000, semilla=SEED):
    """Prueba de White de homocedasticidad sobre una submuestra (regresores: X, X² y productos cruzados)."""
    from statsmodels.stats.diagnostic import het_white
    idx = np.random.default_rng(semilla).choice(len(residuos), min(n, len(residuos)), replace=False)
    Xs = np.column_stack([np.ones(len(idx)), (X[idx] - X[idx].mean(0)) / (X[idx].std(0) + 1e-12)])
    lm, p_lm, f, p_f = het_white(residuos[idx], Xs)
    return {"estadistico_LM": float(lm), "p_valor": float(p_lm)}


def pruebas_independencia(est, t, residuos, max_obs=2000, rezagos=24, min_obs=500):
    """BDS y Ljung-Box estación por estación sobre la serie horaria de residuos.

    BDS (dimensión de inmersión 2, ε = 0.7·σ) detecta dependencia no lineal; Ljung-Box (24 rezagos), la
    autocorrelación lineal. Se usan las últimas ``max_obs`` horas observadas de cada estación (BDS crece como
    n²) y solo estaciones con al menos ``min_obs`` horas.

    Returns
    -------
    pandas.DataFrame
        Una fila por estación con los p-valores de ambas pruebas.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import bds
    df = pd.DataFrame({"est": est, "t": t, "r": residuos}).sort_values(["est", "t"])
    filas = []
    for e, g in df.groupby("est"):
        r = g.r.to_numpy()[-max_obs:]
        if len(r) < min_obs:
            continue
        _, p_bds = bds(r, max_dim=2, epsilon=0.7 * r.std())
        p_lb = acorr_ljungbox(r, lags=[rezagos]).lb_pvalue.iloc[0]
        filas.append({"est": e, "n": len(r), "p_BDS": float(np.ravel(p_bds)[0]), "p_LjungBox": float(p_lb)})
    return pd.DataFrame(filas).set_index("est")


# ------------------------------------------------------------------ comparación de pronósticos

def _var_hac(d, rezagos):
    """Varianza de largo plazo de Newey-West (núcleo de Bartlett)."""
    d = d - d.mean()
    T = len(d)
    v = d @ d / T
    for k in range(1, rezagos + 1):
        v += 2 * (1 - k / (rezagos + 1)) * (d[k:] @ d[:-k]) / T
    return v


def diebold_mariano(l_a, l_b, h=2):
    """Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold (1997).

    Parameters
    ----------
    l_a, l_b : array-like
        Pérdidas diarias de los modelos A y B (mismos días).
    h : int
        Horizonte en días. El pronóstico a 24 h hace que el error de un día se superponga con el del
        siguiente, por eso ``h = 2`` y la varianza usa ``h - 1`` rezagos.

    Returns
    -------
    dict con el estadístico corregido (negativo favorece a A) y el p-valor bilateral con t de Student (T − 1).
    """
    d = np.asarray(l_a) - np.asarray(l_b)
    T = len(d)
    dm = d.mean() / np.sqrt(_var_hac(d, h - 1) / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    est = dm * hln
    return {"DM_HLN": float(est), "p_valor": float(2 * stats.t.sf(abs(est), T - 1)), "diferencia_media": float(d.mean())}


def clark_west(y_dia_err_r, y_dia_err_g, dif_pred2):
    """Clark-West (2007) para modelos anidados: H0 igual capacidad predictiva, H1 el modelo general es mejor.

    Parameters
    ----------
    y_dia_err_r, y_dia_err_g : array-like
        Error cuadrático medio diario del modelo restringido y del general.
    dif_pred2 : array-like
        Media diaria de (ŷ_r − ŷ_g)², el término que corrige el sesgo del DM en modelos anidados.
    """
    f = np.asarray(y_dia_err_r) - (np.asarray(y_dia_err_g) - np.asarray(dif_pred2))
    T = len(f)
    est = f.mean() / np.sqrt(_var_hac(f, 1) / T)
    return {"CW": float(est), "p_valor": float(stats.norm.sf(est))}


def giacomini_white(l_a, l_b):
    """Prueba condicional de Giacomini-White (2006) con instrumentos (1, d_{t−1}).

    Pregunta si la diferencia de pérdidas es predecible con lo observado el día anterior; rechazar indica que
    un modelo es mejor al menos en algunos estados (por ejemplo, después de días con errores grandes).
    """
    d = np.asarray(l_a) - np.asarray(l_b)
    Z = np.column_stack([np.ones(len(d) - 1), d[:-1]])
    zd = Z * d[1:, None]
    T, q = zd.shape
    zbar = zd.mean(0)
    omega = np.cov(zd, rowvar=False, bias=True)
    est = T * zbar @ np.linalg.solve(omega, zbar)
    return {"GW": float(est), "gl": q, "p_valor": float(stats.chi2.sf(est, q)), "signo_medio": float(np.sign(d.mean()))}


def mcs(perdidas, alfa=0.10, bloque=7, reps=2000, semilla=SEED):
    """Model Confidence Set (Hansen, Lunde y Nason, 2011) con bootstrap estacionario.

    Parameters
    ----------
    perdidas : pandas.DataFrame
        Pérdida diaria (filas) de cada modelo (columnas).

    Returns
    -------
    pandas.DataFrame con el p-valor MCS de cada modelo y si queda en el conjunto de confianza.
    """
    from arch.bootstrap import MCS
    m = MCS(perdidas, size=alfa, reps=reps, block_size=bloque, method="R", bootstrap="stationary", seed=semilla)
    m.compute()
    out = m.pvalues.rename(columns={"Pvalue": "p_MCS"})
    out["en el conjunto"] = out.index.isin(m.included)
    out["pérdida media"] = perdidas.mean()[out.index]
    return out.sort_values("pérdida media")


def spa(perdida_referencia, perdidas, bloque=7, reps=2000, semilla=SEED):
    """Reality Check (White, 2000) y SPA (Hansen, 2005) con bootstrap estacionario.

    H0: ningún modelo de ``perdidas`` tiene menor pérdida esperada que la referencia. Controla el *data
    snooping*: el p-valor tiene en cuenta que se probaron varios modelos y se mira el mejor.

    Parameters
    ----------
    perdida_referencia : array-like
        Pérdida diaria del modelo de referencia (benchmark).
    perdidas : pandas.DataFrame
        Pérdida diaria (filas) de los modelos alternativos (columnas), mismos días.

    Returns
    -------
    dict con los p-valores del SPA consistente de Hansen, su cota inferior y la cota superior, que
    corresponde al Reality Check de White.
    """
    from arch.bootstrap import SPA
    s = SPA(np.asarray(perdida_referencia), np.asarray(perdidas), block_size=bloque, reps=reps,
            bootstrap="stationary", seed=semilla)
    s.compute()
    p = s.pvalues
    return {"p_SPA (Hansen)": float(p["consistent"]), "p_SPA inferior": float(p["lower"]),
            "p_Reality Check (White)": float(p["upper"])}


def ic_bootstrap_estacionario(d, bloque=7, reps=2000, semilla=SEED):
    """IC 95 % de la media de una serie diaria con bootstrap estacionario (Politis y Romano, 1994)."""
    from arch.bootstrap import StationaryBootstrap
    bs = StationaryBootstrap(bloque, np.asarray(d), seed=semilla)
    lo, hi = bs.conf_int(np.mean, reps=reps, method="percentile").ravel()
    return float(lo), float(hi)


def friedman_nemenyi(tabla):
    """Friedman y Nemenyi sobre una tabla bloques × modelos (menor es mejor).

    Returns
    -------
    dict con el estadístico y p-valor de Friedman, los rangos promedio, la diferencia crítica (α = 0.05) y la
    matriz de p-valores de Nemenyi.
    """
    import scikit_posthocs as sp
    chi, p = stats.friedmanchisquare(*[tabla[c].to_numpy() for c in tabla])
    N, k = tabla.shape
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    return {"chi2": float(chi), "gl": k - 1, "p_valor": float(p), "rangos": tabla.rank(axis=1).mean().sort_values(),
            "CD": float(q * np.sqrt(k * (k + 1) / (6 * N))), "N": N,
            "nemenyi": sp.posthoc_nemenyi_friedman(tabla.to_numpy()).set_axis(tabla.columns, axis=0).set_axis(tabla.columns, axis=1)}


def delta_cliff(a, b):
    """Delta de Cliff: P(a > b) − P(a < b). Con pérdidas, un valor negativo favorece a ``a``."""
    a, b = np.asarray(a), np.asarray(b)
    mayor = (a[:, None] > b[None, :]).mean()
    menor = (a[:, None] < b[None, :]).mean()
    return float(mayor - menor)


def magnitud_cliff(d):
    d = abs(d)
    return "despreciable" if d < 0.147 else "pequeño" if d < 0.33 else "mediano" if d < 0.474 else "grande"


def d_cohen(l_a, l_b):
    """d de Cohen para diferencias pareadas de pérdida: media de la diferencia / su desviación estándar."""
    d = np.asarray(l_a) - np.asarray(l_b)
    return float(d.mean() / d.std(ddof=1))
