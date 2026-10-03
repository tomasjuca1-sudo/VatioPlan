"""
datos.py — Adquisición automática de datos públicos.

Fuentes (todas gratuitas y sin API key):
  1. SIMEM (XM)  — Precio de bolsa nacional horario, conjunto EC6945.
                   GET https://www.simem.co/backend-files/api/PublicData
  2. API XM       — Variables hidrológicas diarias (volumen útil de embalses, aportes).
                   POST https://servapibi.xm.com.co/daily
  3. NOAA CPC     — Índice ONI (El Niño / La Niña), mensual.
  4. Festivos de Colombia — librería `holidays`.

Si alguna fuente no responde, el pipeline sigue con las demás. Solo si NO hay
precio real disponible (p. ej. sin internet) se generan datos SIMULADOS para poder
probar el código; en ese caso todo se marca con fuente = "SIMULADO".
"""
from __future__ import annotations

import io
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SIMEM_URL = "https://www.simem.co/backend-files/api/PublicData"
XM_URL = "https://servapibi.xm.com.co/{granularidad}"
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Orden real de versiones de liquidación en SIMEM (de la primera a la definitiva)
ORDEN_VERSIONES = ["TX1", "TX2", "TXR", "TXF"] + [f"TX{i}" for i in range(3, 10)]
RANGO_VERSION = {v: i for i, v in enumerate(ORDEN_VERSIONES)}

TIMEOUT = 60


def _bloques(inicio: date, fin: date, dias: int):
    """Divide [inicio, fin] en bloques de `dias` días (las APIs limitan el rango)."""
    actual = inicio
    while actual <= fin:
        tope = min(actual + timedelta(days=dias - 1), fin)
        yield actual, tope
        actual = tope + timedelta(days=1)


# ---------------------------------------------------------------------------
# 1. Precio de bolsa horario (SIMEM)
# ---------------------------------------------------------------------------
def descargar_precio_bolsa(inicio: date, fin: date, verbose: bool = True) -> pd.Series:
    """Precio de bolsa nacional horario (COP/kWh), quedándose con la versión más reciente
    de liquidación de cada hora. Devuelve una Serie indexada por fecha-hora."""
    registros = []
    for a, b in _bloques(inicio, fin, 30):
        params = {"startdate": a.isoformat(), "enddate": b.isoformat(), "datasetId": "EC6945"}
        for intento in range(3):
            try:
                r = requests.get(SIMEM_URL, params=params, timeout=TIMEOUT)
                r.raise_for_status()
                recs = r.json().get("result", {}).get("records", [])
                registros.extend(recs)
                if verbose:
                    print(f"  SIMEM {a} → {b}: {len(recs)} registros")
                break
            except Exception as e:  # reintento simple
                if intento == 2:
                    raise RuntimeError(f"SIMEM falló para {a}–{b}: {e}") from e
                time.sleep(2 * (intento + 1))

    if not registros:
        raise RuntimeError("SIMEM no devolvió registros")

    df = pd.DataFrame(registros)
    df = df[df["CodigoVariable"] == "PB_Nal"].copy()          # solo precio nacional
    df["FechaHora"] = pd.to_datetime(df["FechaHora"])
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
    df["rango"] = df["Version"].map(RANGO_VERSION).fillna(-1)
    df = df.sort_values(["FechaHora", "rango"]).drop_duplicates("FechaHora", keep="last")
    serie = df.set_index("FechaHora")["Valor"].sort_index()
    serie.name = "precio"
    return serie


# ---------------------------------------------------------------------------
# 2. Variables hidrológicas diarias (API XM)
# ---------------------------------------------------------------------------
def descargar_xm(metrica: str, inicio: date, fin: date, entidad: str = "Sistema",
                 granularidad: str = "daily", verbose: bool = True) -> pd.Series:
    """Descarga una métrica diaria de la API de XM (ej. 'PorcVoluUtilDiar', 'AporEner')."""
    filas = []
    for a, b in _bloques(inicio, fin, 30):
        cuerpo = {"MetricId": metrica, "StartDate": a.isoformat(), "EndDate": b.isoformat(),
                  "Entity": entidad, "Filter": []}
        r = requests.post(XM_URL.format(granularidad=granularidad), json=cuerpo, timeout=TIMEOUT)
        r.raise_for_status()
        for item in r.json().get("Items", []):
            for ent in item.get("DailyEntities", []):
                valor = ent.get("Values", {}).get("Value")
                filas.append({"fecha": item.get("Date"), "valor": valor})
                if len(filas) == 1:
                    print(f"  XM {metrica} primera entidad cruda: {json.dumps(ent, ensure_ascii=False)[:400]}")
        if verbose:
            print(f"  XM {metrica} {a} → {b}: ok")
    s = pd.DataFrame(filas)
    muestra = s.head(3).to_dict("records")
    s["fecha"] = pd.to_datetime(s["fecha"].astype(str).str[:10])   # solo AAAA-MM-DD, sin hora ni zona
    s["valor"] = pd.to_numeric(s["valor"], errors="coerce")
    if verbose or s["valor"].notna().sum() == 0:
        print(f"  XM {metrica}: {s['valor'].notna().sum()} de {len(s)} filas con valor. Muestra cruda: {muestra}")
    return s.groupby("fecha")["valor"].mean().rename(metrica)


# ---------------------------------------------------------------------------
# 3. Índice ONI (NOAA)
# ---------------------------------------------------------------------------
_TEMPORADA_A_MES = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def descargar_oni() -> pd.Series:
    """ONI mensual (anomalía de temperatura del Pacífico). >0.5 = El Niño (sequía en Colombia)."""
    r = requests.get(ONI_URL, timeout=TIMEOUT)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=r"\s+")
    df["mes"] = df["SEAS"].map(_TEMPORADA_A_MES)
    df["fecha"] = pd.to_datetime(dict(year=df["YR"], month=df["mes"], day=1))
    return df.set_index("fecha")["ANOM"].rename("oni")


# ---------------------------------------------------------------------------
# 4. Datos simulados (solo para probar sin internet)
# ---------------------------------------------------------------------------
def generar_datos_simulados(inicio: date, fin: date, semilla: int = 7) -> pd.DataFrame:
    """Serie horaria SINTÉTICA con la forma típica del mercado colombiano:
    pico nocturno 18–21 h, valle de madrugada, efecto hidrología/El Niño y domingos.
    NO son datos reales: sirve para ejecutar el pipeline sin conexión."""
    rng = np.random.default_rng(semilla)
    idx = pd.date_range(inicio, pd.Timestamp(fin) + pd.Timedelta(hours=23), freq="h")
    dias = pd.date_range(inicio, fin, freq="D")
    t = np.arange(len(dias))
    # hidrología: dos temporadas de lluvia al año + episodio tipo El Niño
    doy = dias.dayofyear.values
    lluvia = 0.5 + 0.25 * np.sin(2 * np.pi * (doy - 60) / 182.5)
    nino = np.exp(-0.5 * ((t - len(t) * 0.35) / 60) ** 2)
    vol = np.clip(70 * lluvia + 20 - 35 * nino + rng.normal(0, 2, len(t)).cumsum() * 0.05, 25, 95)
    aportes = np.clip(200e6 * lluvia * (1 - 0.5 * nino) * rng.lognormal(0, 0.15, len(t)), 5e7, None)
    oni = 0.2 + 1.8 * nino - 0.6 * (1 - nino) * np.sin(t / 200)
    nivel = 250 + 6 * (80 - vol) + 250 * nino                       # COP/kWh
    diario = pd.DataFrame({"vol_util_pct": vol, "aportes_gwh": aportes / 1e6, "oni": oni,
                           "nivel": nivel}, index=dias)

    forma = np.array([0.85, 0.82, 0.80, 0.80, 0.82, 0.90, 0.95, 0.97, 0.95, 0.92, 0.88, 0.85,
                      0.86, 0.90, 0.98, 1.05, 1.10, 1.25, 1.75, 1.85, 1.55, 1.30, 1.10, 0.95])
    df = pd.DataFrame(index=idx)
    d = diario.reindex(idx.normalize()).set_axis(idx)
    domingo = np.where(idx.dayofweek == 6, 0.9, 1.0)
    ruido = rng.lognormal(0, 0.08, len(idx))
    picos = np.where(rng.random(len(idx)) < 0.01, rng.uniform(1.3, 2.2, len(idx)), 1.0)
    df["precio"] = d["nivel"].values * forma[idx.hour] * domingo * ruido * picos
    df["vol_util_pct"] = d["vol_util_pct"].values
    df["aportes_gwh"] = d["aportes_gwh"].values
    df["oni"] = d["oni"].values
    return df


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def cargar_datos(inicio: date | str = "2023-01-01", fin: date | str | None = None,
                 carpeta_cache: str | Path = "data", permitir_simulado: bool = True,
                 verbose: bool = True) -> pd.DataFrame:
    """Devuelve un DataFrame horario con: precio, vol_util_pct, aportes_gwh, oni.
    El atributo df.attrs['fuente'] indica 'REAL' o 'SIMULADO'.
    Usa caché en disco: solo descarga los días que faltan."""
    inicio = pd.Timestamp(inicio).date()
    fin = pd.Timestamp(fin).date() if fin else date.today()
    cache = Path(carpeta_cache)
    cache.mkdir(parents=True, exist_ok=True)
    archivo = cache / "precio_bolsa.csv"

    # --- precio (incremental con caché) ---
    precio = None
    try:
        previo = pd.read_csv(archivo, index_col=0, parse_dates=True)["precio"] if archivo.exists() else None
        desde = inicio
        if previo is not None and len(previo):
            # re-descarga los últimos 10 días porque XM revisa los precios
            desde = max(inicio, (previo.index.max() - pd.Timedelta(days=10)).date())
        if verbose:
            print(f"Descargando precio de bolsa (SIMEM) {desde} → {fin} ...")
        nuevo = descargar_precio_bolsa(desde, fin, verbose=verbose)
        precio = nuevo if previo is None else pd.concat([previo[previo.index < pd.Timestamp(desde)], nuevo])
        precio = precio[~precio.index.duplicated(keep="last")].sort_index()
        precio.to_frame().to_csv(archivo)
    except Exception as e:
        if verbose:
            print(f"⚠️  No se pudo descargar el precio: {e}")
        if archivo.exists():
            precio = pd.read_csv(archivo, index_col=0, parse_dates=True)["precio"]
            if verbose:
                print("   Usando caché local.")

    if precio is None or precio.empty:
        if not permitir_simulado:
            raise RuntimeError("Sin datos de precio y permitir_simulado=False")
        if verbose:
            print("⚠️  SIN CONEXIÓN A SIMEM → se usan DATOS SIMULADOS (solo para pruebas).")
        df = generar_datos_simulados(inicio, fin - timedelta(days=3))
        df.attrs["fuente"] = "SIMULADO"
        return df

    precio = precio[precio.index >= pd.Timestamp(inicio)]
    idx = pd.date_range(precio.index.min(), precio.index.max(), freq="h")
    df = precio.reindex(idx).interpolate(limit=3).to_frame("precio")

    # --- variables hidrológicas (opcionales) ---
    dias_ini, dias_fin = idx.min().date(), idx.max().date()
    for metrica, col, escala in [("PorcVoluUtilDiar", "vol_util_pct", 100.0),
                                 ("AporEner", "aportes_gwh", 1e-6)]:
        f_cache = cache / f"{metrica}.csv"
        try:
            s = descargar_xm(metrica, dias_ini, dias_fin, verbose=verbose) * escala
            s.to_frame().to_csv(f_cache)
        except Exception as e:
            if f_cache.exists():
                s = pd.read_csv(f_cache, index_col=0, parse_dates=True).iloc[:, 0]
            else:
                if verbose:
                    print(f"⚠️  {metrica} no disponible ({e}); el modelo seguirá sin esta variable.")
                continue
        df[col] = s.reindex(idx.normalize()).values
    # PorcVoluUtilDiar viene en fracción (0–1) en algunos años; se normaliza a %
    if "vol_util_pct" in df and df["vol_util_pct"].max() <= 1.5:
        df["vol_util_pct"] *= 100

    try:
        oni = descargar_oni()
        oni.to_frame().to_csv(cache / "oni.csv")
    except Exception:
        oni = pd.read_csv(cache / "oni.csv", index_col=0, parse_dates=True)["oni"] if (cache / "oni.csv").exists() else None
    if oni is not None:
        mensual = idx.to_period("M").to_timestamp()
        df["oni"] = oni.reindex(mensual).ffill().values

    # Una variable que llega 100% vacía no sirve y rompe el entrenamiento en scikit-learn 1.9+.
    # Se descarta y se avisa, para que el modelo siga con las variables que sí llegaron.
    for col in ["vol_util_pct", "aportes_gwh", "oni"]:
        if col in df and df[col].isna().all():
            print(f"⚠️  La variable '{col}' llegó vacía; el modelo seguirá sin ella.")
            del df[col]

    df.attrs["fuente"] = "REAL"
    if verbose:
        print(f"✅ Datos reales: {df.index.min()} → {df.index.max()} ({len(df):,} horas)")
    return df
