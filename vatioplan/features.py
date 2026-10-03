"""
features.py — Variables de entrada para el modelo.

Regla clave para evitar FUGA DE DATOS: el precio se publica con ~3 días de retraso y
queremos planear varios días hacia adelante. Por eso TODAS las variables usan
información de al menos 7 días atrás (168 h). Así un solo modelo pronostica
cualquier hora de la próxima semana sin "ver el futuro".

No se usa la demanda o generación del mismo momento: se deciden en el mismo
despacho que el precio (correlación engañosa, ver diapositiva de correlaciones espurias).
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

HORIZONTE_H = 168  # 7 días


def festivos_co(indice: pd.DatetimeIndex) -> np.ndarray:
    fest = holidays.country_holidays("CO", years=range(indice.year.min(), indice.year.max() + 1))
    return np.array([d in fest for d in indice.date], dtype=int)


def construir_features(df: pd.DataFrame, horizonte: int = HORIZONTE_H) -> pd.DataFrame:
    """Recibe el DataFrame horario (con NaN en 'precio' para las horas futuras)
    y devuelve la matriz de variables X (misma longitud)."""
    p = df["precio"]
    X = pd.DataFrame(index=df.index)
    X["hora"] = df.index.hour
    X["dia_semana"] = df.index.dayofweek
    X["mes"] = df.index.month
    X["festivo"] = festivos_co(df.index)
    X["dia_no_laboral"] = ((X["dia_semana"] == 6) | (X["festivo"] == 1)).astype(int)

    # Rezagos del precio (≥ horizonte)
    X["precio_lag_168h"] = p.shift(horizonte)                 # misma hora, hace 7 días
    X["precio_lag_336h"] = p.shift(horizonte + 168)           # misma hora, hace 14 días
    media_24 = p.rolling(24, min_periods=12).mean()
    X["media_dia_hace_7d"] = media_24.shift(horizonte)
    X["media_semana_previa"] = p.rolling(168, min_periods=84).mean().shift(horizonte)
    # forma relativa de la hora (precio de esa hora / promedio del día), hace 7 días
    X["forma_hora_hace_7d"] = (p / media_24.reindex(p.index)).shift(horizonte)
    X["volatilidad_semana_previa"] = p.rolling(168, min_periods=84).std().shift(horizonte)

    # Hidrología y clima (ya son diarios/mensuales, se rezagan igual)
    if "vol_util_pct" in df:
        X["embalses_pct_hace_7d"] = df["vol_util_pct"].shift(horizonte)
        X["embalses_cambio_30d"] = df["vol_util_pct"].shift(horizonte) - df["vol_util_pct"].shift(horizonte + 720)
    if "aportes_gwh" in df:
        X["aportes_media_7d"] = df["aportes_gwh"].rolling(168, min_periods=24).mean().shift(horizonte)
    if "oni" in df:
        X["oni_hace_2m"] = df["oni"].shift(24 * 60)
    return X


def extender_futuro(df: pd.DataFrame, horas: int = HORIZONTE_H) -> pd.DataFrame:
    """Agrega `horas` filas vacías al final (las horas a pronosticar)."""
    futuro = pd.date_range(df.index.max() + pd.Timedelta(hours=1), periods=horas, freq="h")
    ext = pd.concat([df, pd.DataFrame(index=futuro)])
    for c in ["vol_util_pct", "aportes_gwh", "oni"]:
        if c in ext:
            ext[c] = ext[c].ffill()
    ext.attrs = df.attrs
    return ext
