"""
planta.py — Programación de cargas flexibles + reglas de ingeniería.

Para cada día se decide a qué horas funciona cada carga flexible, minimizando
    costo = Σ_h  precio(h) · potencia · x(carga, h)
sujeto a las REGLAS DE INGENIERÍA:
  R1  Cada carga cumple sus horas diarias requeridas (producción no se sacrifica).
  R2  Solo opera en su ventana permitida (turnos, disponibilidad de operario).
  R3  Las cargas continuas (horno) operan en un único bloque sin interrupción.
  R4  La potencia total de la planta nunca supera la potencia máxima contratada.
Se resuelve como un problema de programación lineal entera (PuLP + solver CBC, gratis).
El plan se valida después con las mismas reglas → "Cumple / No cumple".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pulp
except ImportError:  # el greedy de respaldo funciona sin PuLP
    pulp = None


@dataclass
class Carga:
    nombre: str
    potencia_kw: float
    horas_dia: int
    continua: bool
    horas_permitidas: list
    inicio_actual: int
    motivo_restriccion: str = ""
    permitidas: set = field(default_factory=set, init=False)

    def __post_init__(self):
        rangos = self.horas_permitidas
        if rangos and not isinstance(rangos[0], list):
            rangos = [rangos]
        self.permitidas = {h for a, b in rangos for h in range(a, b + 1)}


@dataclass
class Planta:
    nombre: str
    potencia_maxima_kw: float
    base: dict
    cargas: list
    exposicion_bolsa: float = 1.0
    descripcion: str = ""

    @classmethod
    def desde_json(cls, ruta: str | Path) -> "Planta":
        c = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(nombre=c["nombre"], potencia_maxima_kw=c["potencia_maxima_kw"],
                   base=c["carga_base_kw"], cargas=[Carga(**x) for x in c["cargas_flexibles"]],
                   exposicion_bolsa=c.get("exposicion_bolsa", 1.0), descripcion=c.get("descripcion", ""))

    def carga_base(self) -> np.ndarray:
        b = self.base
        return np.array([b["turno_operado"] if b["inicio_turno"] <= h < b["fin_turno"] else b["noche"]
                         for h in range(24)], dtype=float)


# ---------------------------------------------------------------------------
# Programas
# ---------------------------------------------------------------------------
def plan_actual(planta: Planta) -> pd.DataFrame:
    """Cómo opera hoy la planta: cada carga arranca a su hora habitual (sin mirar el precio)."""
    plan = pd.DataFrame(0, index=range(24), columns=[c.nombre for c in planta.cargas])
    for c in planta.cargas:
        for k in range(c.horas_dia):
            plan.loc[(c.inicio_actual + k) % 24, c.nombre] = 1
    return plan


def optimizar_dia(precios: np.ndarray, planta: Planta) -> pd.DataFrame:
    """Plan óptimo de un día (24 precios) con PuLP. Si PuLP no está, usa el greedy."""
    if pulp is None:
        return _greedy_dia(precios, planta)
    H = range(24)
    base = planta.carga_base()
    prob = pulp.LpProblem("programacion_cargas", pulp.LpMinimize)
    x = {(c.nombre, h): pulp.LpVariable(f"x_{i}_{h}", cat="Binary")
         for i, c in enumerate(planta.cargas) for h in H}
    prob += pulp.lpSum(precios[h] * c.potencia_kw * x[c.nombre, h] for c in planta.cargas for h in H)

    for i, c in enumerate(planta.cargas):
        prob += pulp.lpSum(x[c.nombre, h] for h in H) == c.horas_dia                         # R1
        for h in H:
            if h not in c.permitidas:
                prob += x[c.nombre, h] == 0                                                  # R2
        if c.continua:                                                                        # R3
            inicios = [s for s in H if all((s + k) in c.permitidas for k in range(c.horas_dia))]
            y = {s: pulp.LpVariable(f"y_{i}_{s}", cat="Binary") for s in inicios}
            prob += pulp.lpSum(y.values()) == 1
            for h in H:
                prob += x[c.nombre, h] == pulp.lpSum(y[s] for s in inicios if s <= h < s + c.horas_dia)
    for h in H:                                                                               # R4
        prob += base[h] + pulp.lpSum(c.potencia_kw * x[c.nombre, h] for c in planta.cargas) \
            <= planta.potencia_maxima_kw

    estado = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[estado] != "Optimal":
        return _greedy_dia(precios, planta)
    plan = pd.DataFrame(0, index=range(24), columns=[c.nombre for c in planta.cargas])
    for (n, h), v in x.items():
        plan.loc[h, n] = int(round(v.value()))
    return plan


def _greedy_dia(precios: np.ndarray, planta: Planta) -> pd.DataFrame:
    """Respaldo sin solver: asigna primero las cargas grandes a sus horas más baratas."""
    base = planta.carga_base().copy()
    plan = pd.DataFrame(0, index=range(24), columns=[c.nombre for c in planta.cargas])
    for c in sorted(planta.cargas, key=lambda c: -c.potencia_kw):
        libre = lambda h: h in c.permitidas and base[h] + c.potencia_kw <= planta.potencia_maxima_kw
        if c.continua:
            opciones = [(sum(precios[s:s + c.horas_dia]), s) for s in range(24 - c.horas_dia + 1)
                        if all(libre(s + k) for k in range(c.horas_dia))]
            horas = range(min(opciones)[1], min(opciones)[1] + c.horas_dia) if opciones else []
        else:
            horas = sorted([h for h in range(24) if libre(h)], key=lambda h: precios[h])[:c.horas_dia]
        for h in horas:
            plan.loc[h, c.nombre] = 1
            base[h] += c.potencia_kw
    return plan


# ---------------------------------------------------------------------------
# Reglas, costos y planes de varios días
# ---------------------------------------------------------------------------
def validar_plan(plan: pd.DataFrame, planta: Planta) -> pd.DataFrame:
    """Verifica las reglas R1–R4. Devuelve una tabla Regla | Carga | Resultado."""
    filas = []
    total = planta.carga_base() + sum(plan[c.nombre].values * c.potencia_kw for c in planta.cargas)
    for c in planta.cargas:
        horas = [h for h in range(24) if plan.loc[h, c.nombre] == 1]
        filas.append(("R1 Horas requeridas", c.nombre, len(horas) == c.horas_dia))
        filas.append(("R2 Ventana permitida", c.nombre, all(h in c.permitidas for h in horas)))
        if c.continua:
            filas.append(("R3 Operación continua", c.nombre,
                          bool(horas) and horas[-1] - horas[0] + 1 == len(horas)))
    filas.append(("R4 Potencia ≤ máxima", "Planta completa", bool((total <= planta.potencia_maxima_kw + 1e-6).all())))
    t = pd.DataFrame(filas, columns=["regla", "carga", "cumple"])
    t["resultado"] = np.where(t["cumple"], "✅ Cumple", "❌ No cumple")
    return t


def costo_plan(plan: pd.DataFrame, precios: np.ndarray, planta: Planta) -> float:
    """Costo en COP de las cargas flexibles (precio en COP/kWh × kWh × exposición a bolsa)."""
    kwh = sum(plan[c.nombre].values * c.potencia_kw for c in planta.cargas)
    return float(np.sum(kwh * precios) * planta.exposicion_bolsa)


def planear_periodo(precios_pronostico: pd.Series, planta: Planta) -> dict:
    """Optimiza cada día completo del pronóstico y compara contra la operación actual."""
    resultados = []
    for dia, g in precios_pronostico.groupby(precios_pronostico.index.date):
        if len(g) < 24:
            continue
        p = g.sort_index().values[:24]
        opt, act = optimizar_dia(p, planta), plan_actual(planta)
        resultados.append({
            "fecha": pd.Timestamp(dia), "precios": p, "plan_optimo": opt, "plan_actual": act,
            "costo_actual": costo_plan(act, p, planta), "costo_optimo": costo_plan(opt, p, planta),
            "validacion": validar_plan(opt, planta)})
    return {"dias": resultados,
            "costo_actual": sum(r["costo_actual"] for r in resultados),
            "costo_optimo": sum(r["costo_optimo"] for r in resultados)}


def backtest(real: pd.Series, pred: pd.Series, planta: Planta) -> pd.DataFrame:
    """Evalúa con precios REALES un plan hecho con el PRONÓSTICO (lo que pasaría en la vida real).
    Compara: operación actual vs. plan con pronóstico vs. plan perfecto (conociendo el precio)."""
    filas = []
    act = plan_actual(planta)
    d = pd.DataFrame({"real": real, "pred": pred}).dropna()
    for dia, g in d.groupby(d.index.date):
        if len(g) != 24:
            continue
        r, p = g["real"].values, g["pred"].values
        filas.append({"fecha": pd.Timestamp(dia),
                      "costo_actual": costo_plan(act, r, planta),
                      "costo_con_pronostico": costo_plan(optimizar_dia(p, planta), r, planta),
                      "costo_perfecto": costo_plan(optimizar_dia(r, planta), r, planta)})
    return pd.DataFrame(filas).set_index("fecha")


def perfil_potencia(plan: pd.DataFrame, planta: Planta) -> pd.DataFrame:
    out = pd.DataFrame({"Carga base": planta.carga_base()})
    for c in planta.cargas:
        out[c.nombre] = plan[c.nombre].values * c.potencia_kw
    return out
