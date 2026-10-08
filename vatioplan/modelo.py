"""
modelo.py — Pronóstico del precio de bolsa con Machine Learning.

Modelo: HistGradientBoostingRegressor (scikit-learn, gratuito, maneja datos faltantes).
Línea base: "ingenuo estacional" = el precio de la misma hora hace 7 días.
Métricas:
  * MAE y MAPE del precio.
  * Acierto de horas baratas: de las 8 horas más baratas reales de cada día,
    ¿cuántas identificó el modelo? (Es lo que importa para programar la planta.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .features import HORIZONTE_H, construir_features, extender_futuro


def _dataset(df: pd.DataFrame):
    X = construir_features(df)
    y = df["precio"]
    ok = y.notna() & X["precio_lag_168h"].notna()
    return X[ok], y[ok]


def nuevo_modelo(semilla: int = 42) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, categorical_features=[0, 1, 2],
        random_state=semilla)


def acierto_horas_baratas(real: pd.Series, pred: pd.Series, k: int = 8) -> float:
    """Promedio diario de la fracción de las k horas más baratas reales que el pronóstico
    también puso entre sus k más baratas."""
    d = pd.DataFrame({"real": real, "pred": pred}).dropna()
    aciertos = []
    for _, g in d.groupby(d.index.date):
        if len(g) < 24:
            continue
        top_r = set(g["real"].nsmallest(k).index)
        top_p = set(g["pred"].nsmallest(k).index)
        aciertos.append(len(top_r & top_p) / k)
    return float(np.mean(aciertos)) if aciertos else float("nan")


def evaluar(df: pd.DataFrame, dias_prueba: int = 90) -> dict:
    """Entrena con todo excepto los últimos `dias_prueba` días y evalúa en ellos."""
    X, y = _dataset(df)
    corte = y.index.max() - pd.Timedelta(days=dias_prueba)
    ent, pru = y.index <= corte, y.index > corte
    y_residuo = y[ent] - X.loc[ent, "precio_lag_168h"]
    modelo = nuevo_modelo().fit(X[ent], y_residuo)
    pred_valores = X.loc[pru, "precio_lag_168h"].values + modelo.predict(X[pru])
    pred = pd.Series(pred_valores, index=y.index[pru])
    base = X.loc[pru, "precio_lag_168h"]
    real = y[pru]

    def mape(r, p):
        return float(np.mean(np.abs((r - p) / r.clip(lower=1))) * 100)

    res = {
        "periodo_prueba": f"{real.index.min():%Y-%m-%d} → {real.index.max():%Y-%m-%d}",
        "horas_entrenamiento": int(ent.sum()),
        "horas_prueba": int(pru.sum()),
        "mae_modelo": float(mean_absolute_error(real, pred)),
        "mae_base": float(mean_absolute_error(real, base)),
        "mape_modelo": mape(real, pred),
        "mape_base": mape(real, base),
        "acierto_horas_baratas_modelo": acierto_horas_baratas(real, pred),
        "acierto_horas_baratas_base": acierto_horas_baratas(real, base),
    }
    res["mejora_mae_pct"] = (1 - res["mae_modelo"] / res["mae_base"]) * 100
    return {"metricas": res, "modelo": modelo, "real": real, "pred": pred, "base": base}


def entrenar(df: pd.DataFrame) -> HistGradientBoostingRegressor:
    X, y = _dataset(df)
    y_residuo = y - X["precio_lag_168h"]
    return nuevo_modelo().fit(X, y_residuo)


def pronosticar(df: pd.DataFrame, modelo, horas: int = HORIZONTE_H) -> pd.Series:
    """Pronóstico de las próximas `horas` horas después del último dato disponible."""
    ext = extender_futuro(df, horas)
    X = construir_features(ext)
    fut = X.index[X.index > df.index.max()]
    pred_valores = X.loc[fut, "precio_lag_168h"].values + modelo.predict(X.loc[fut])
    return pd.Series(pred_valores, index=fut, name="precio_pronostico")


def importancia_variables(modelo, df: pd.DataFrame, n_muestras: int = 3000) -> pd.Series:
    """Importancia por permutación (qué variables usa más el modelo)."""
    from sklearn.inspection import permutation_importance
    X, y = X.tail(n_muestras), y.tail(n_muestras)
    y = y - X["precio_lag_168h"]
    r = permutation_importance(modelo, X, y, n_repeats=5, random_state=0,
                               scoring="neg_mean_absolute_error")
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
