"""
modelo.py
Pronostico del precio de bolsa con Machine Learning.

Modelo principal: gradient boosting (HistGradientBoostingRegressor de scikit-learn) que
corrige el promedio de las ultimas 4 semanas.
  1. Ancla: promedio del precio de la misma hora hace 1, 2, 3 y 4 semanas. Trae el nivel
     y la forma del dia del regimen actual del mercado.
  2. Correccion: el gradient boosting aprende cuanto se desvia el precio real del ancla
     segun la hora, el dia de la semana y los festivos.
  Pronostico = ancla + correccion.

Por que este diseno (ver diagnostico.py y diagnostico_hmm.py):
  * Un modelo que pronostica el precio absoluto aprende niveles de anios pasados
    (octubre de 2024 costo 1,529 COP/kWh y octubre de 2025, 171) y falla cuando cambia el regimen.
  * Si el modelo recibe los rezagos del precio, aprende a deshacer el ancla. Por eso la
    correccion usa solo variables de calendario.

Candidatos de comparacion (se evaluan en el mismo periodo de prueba):
  * Ingenuo: precio de la misma hora hace 7 dias.
  * Promedio 4 semanas sin correccion.
  * GB sobre el ingenuo con rezagos de precio (version anterior).
  * GB del precio absoluto con todas las variables (version original).

Metricas: MAE, MAPE y acierto de horas baratas (de las 8 horas mas baratas reales de
cada dia, cuantas identifico el pronostico). El ahorro en backtest se calcula en pipeline.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .features import HORIZONTE_H, construir_features, extender_futuro

SEMANAS_PROMEDIO = 4
COLUMNAS_MODELO = ["hora", "dia_semana", "festivo", "dia_no_laboral"]
COLUMNAS_CATEGORICAS = ["hora", "dia_semana", "mes"]
COLUMNAS_REZAGOS = ["precio_lag_168h", "precio_lag_336h", "media_dia_hace_7d", "media_semana_previa",
                    "forma_hora_hace_7d", "volatilidad_semana_previa"]

NOMBRE_PRINCIPAL = "GB sobre promedio 4 semanas (modelo principal)"
NOMBRE_PROMEDIO = "Promedio 4 semanas"
NOMBRE_INGENUO = "Ingenuo (hace 7 días)"
NOMBRE_GB_INGENUO = "GB sobre ingenuo con rezagos (versión anterior)"
NOMBRE_GB_ABSOLUTO = "GB precio absoluto, todas las variables (versión original)"


def promedio_semanas(precio: pd.Series, semanas: int = SEMANAS_PROMEDIO) -> pd.Series:
    """Promedio del precio de la misma hora hace 1, 2, ..., `semanas` semanas."""
    rezagos = []
    for k in range(1, semanas + 1):
        rezagos.append(precio.shift(HORIZONTE_H * k))
    return pd.concat(rezagos, axis=1).mean(axis=1)


def _dataset(df: pd.DataFrame):
    """X con todas las variables, y = precio real, ancla = promedio de 4 semanas."""
    X = construir_features(df)
    y = df["precio"]
    ancla = promedio_semanas(y)
    filas_ok = y.notna() & ancla.notna() & X["precio_lag_168h"].notna()
    return X[filas_ok], y[filas_ok], ancla[filas_ok]


def nuevo_modelo(columnas: list[str] = COLUMNAS_MODELO, semilla: int = 42) -> HistGradientBoostingRegressor:
    categoricas = []
    for i, columna in enumerate(columnas):
        if columna in COLUMNAS_CATEGORICAS:
            categoricas.append(i)
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, categorical_features=categoricas,
        random_state=semilla)


def acierto_horas_baratas(real: pd.Series, pred: pd.Series, k: int = 8) -> float:
    """Promedio diario de la fraccion de las k horas mas baratas reales que el pronostico
    tambien puso entre sus k mas baratas."""
    d = pd.DataFrame({"real": real, "pred": pred}).dropna()
    aciertos = []
    for _, g in d.groupby(d.index.date):
        if len(g) < 24:
            continue
        top_r = set(g["real"].nsmallest(k).index)
        top_p = set(g["pred"].nsmallest(k).index)
        aciertos.append(len(top_r & top_p) / k)
    return float(np.mean(aciertos)) if aciertos else float("nan")


def mape(real: pd.Series, pred: pd.Series) -> float:
    return float(np.mean(np.abs((real - pred) / real.clip(lower=1))) * 100)


def evaluar(df: pd.DataFrame, dias_prueba: int = 90) -> dict:
    """Entrena con todo excepto los ultimos `dias_prueba` dias y evalua en ellos
    el modelo principal y los candidatos de comparacion."""
    X, y, ancla = _dataset(df)
    corte = y.index.max() - pd.Timedelta(days=dias_prueba)
    ent = y.index <= corte
    pru = y.index > corte
    real = y[pru]
    ingenuo = X["precio_lag_168h"]

    # --- Modelo principal: GB que corrige el promedio de 4 semanas ---
    modelo = nuevo_modelo(COLUMNAS_MODELO).fit(X.loc[ent, COLUMNAS_MODELO], y[ent] - ancla[ent])
    pred = pd.Series(ancla[pru].values + modelo.predict(X.loc[pru, COLUMNAS_MODELO]), index=real.index)

    # --- Candidatos de comparacion ---
    columnas_gb_ingenuo = ["hora", "dia_semana", "festivo", "dia_no_laboral"] + COLUMNAS_REZAGOS
    modelo_gb_ingenuo = nuevo_modelo(columnas_gb_ingenuo).fit(
        X.loc[ent, columnas_gb_ingenuo], y[ent] - ingenuo[ent])
    pred_gb_ingenuo = ingenuo[pru].values + modelo_gb_ingenuo.predict(X.loc[pru, columnas_gb_ingenuo])

    columnas_todas = list(X.columns)
    modelo_absoluto = nuevo_modelo(columnas_todas).fit(X.loc[ent, columnas_todas], y[ent])
    pred_absoluto = modelo_absoluto.predict(X.loc[pru, columnas_todas])

    candidatos = {
        NOMBRE_PRINCIPAL: pred,
        NOMBRE_PROMEDIO: ancla[pru],
        NOMBRE_INGENUO: ingenuo[pru],
        NOMBRE_GB_INGENUO: pd.Series(pred_gb_ingenuo, index=real.index),
        NOMBRE_GB_ABSOLUTO: pd.Series(pred_absoluto, index=real.index),
    }

    base = ingenuo[pru]
    res = {
        "periodo_prueba": f"{real.index.min():%Y-%m-%d} a {real.index.max():%Y-%m-%d}",
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
    return {"metricas": res, "modelo": modelo, "real": real, "pred": pred, "base": base,
            "candidatos": candidatos}


def entrenar(df: pd.DataFrame) -> HistGradientBoostingRegressor:
    X, y, ancla = _dataset(df)
    return nuevo_modelo(COLUMNAS_MODELO).fit(X[COLUMNAS_MODELO], y - ancla)


def pronosticar(df: pd.DataFrame, modelo, horas: int = HORIZONTE_H) -> pd.Series:
    """Pronostico de las proximas `horas` horas despues del ultimo dato disponible."""
    ext = extender_futuro(df, horas)
    X = construir_features(ext)
    ancla = promedio_semanas(ext["precio"])
    fut = X.index[X.index > df.index.max()]
    pred_valores = ancla[fut].values + modelo.predict(X.loc[fut, COLUMNAS_MODELO])
    return pd.Series(pred_valores, index=fut, name="precio_pronostico")


def importancia_variables(modelo, df: pd.DataFrame, n_muestras: int = 3000) -> pd.Series:
    """Importancia por permutacion de las variables del modelo principal."""
    from sklearn.inspection import permutation_importance
    X, y, ancla = _dataset(df)
    X = X[COLUMNAS_MODELO].tail(n_muestras)
    y_residuo = (y - ancla).tail(n_muestras)
    r = permutation_importance(modelo, X, y_residuo, n_repeats=5, random_state=0,
                               scoring="neg_mean_absolute_error")
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)

