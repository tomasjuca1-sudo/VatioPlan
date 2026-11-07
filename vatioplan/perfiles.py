"""
perfiles.py: Los dos tipos de usuario de VatioPlan con una misma interfaz.

  * Fábrica manufacturera (PerfilFabrica): mueve las horas de operación de equipos
    flexibles (molinos, chillers, bombeo...). Reglas R1 a R5 de planta.py.
  * Flota eléctrica (PerfilFlota): mueve la carga de los vehículos mientras están en el
    patio. Reglas R1 a R4 de flotas.py.

Ambos usan el MISMO pronóstico del precio de bolsa. Lo que cambia es la ventana de
planeación (la fábrica planea de 00:00 a 23:00; la flota, por defecto, de 12:00 a 12:00
para que la noche quede completa), el plan de referencia (operación actual vs. carga
inmediata) y las reglas. Con esta interfaz común, el backtest, la comparación de modelos
y el resumen para la IA se escriben una sola vez.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import flotas as fl
from . import planta as pl
from .modelo import NOMBRE_PRINCIPAL, acierto_horas_baratas, mape

RAIZ = Path(__file__).resolve().parent.parent
CONFIG_POR_DEFECTO = {
    "fabrica": RAIZ / "config" / "fabrica_argos_rioclaro.json",
    "flota": RAIZ / "config" / "flota_fontibon_refugio.json",
}
ETIQUETAS = {"fabrica": "🏭 Fábrica manufacturera", "flota": "🚌 Flota eléctrica"}


def _rangos(posiciones: list[int], hora_inicio: int) -> str:
    """Posiciones de la ventana → '22:00 a 02:00, 13:00 a 14:00' en hora de reloj."""
    if not posiciones:
        return "no opera"
    bloques, ini = [], None
    posiciones = sorted(posiciones)
    for i, h in enumerate(posiciones):
        ini = h if ini is None else ini
        if i == len(posiciones) - 1 or posiciones[i + 1] != h + 1:
            bloques.append(f"{(ini + hora_inicio) % 24:02d}:00 a {(h + 1 + hora_inicio) % 24:02d}:00")
            ini = None
    return ", ".join(bloques)


class PerfilFabrica:
    tipo = "fabrica"
    etiqueta = ETIQUETAS["fabrica"]
    nombre_referencia = "Operación actual"
    unidad_elemento = "equipo"
    hora_inicio = 0

    def __init__(self, planta: pl.Planta):
        self.planta = planta

    # --- datos generales ---
    @property
    def nombre(self): return self.planta.nombre
    @property
    def descripcion(self): return self.planta.descripcion
    @property
    def fuentes(self): return self.planta.fuentes
    @property
    def supuestos(self): return self.planta.supuestos
    @property
    def exposicion_bolsa(self): return self.planta.exposicion_bolsa
    @exposicion_bolsa.setter
    def exposicion_bolsa(self, v): self.planta.exposicion_bolsa = v
    @property
    def limite_kw(self): return self.planta.potencia_maxima_kw
    @limite_kw.setter
    def limite_kw(self, v): self.planta.potencia_maxima_kw = v
    @property
    def elementos(self): return [c.nombre for c in self.planta.cargas]

    # --- planes ---
    def plan_referencia(self) -> pd.DataFrame:
        return pl.plan_actual(self.planta)

    def optimizar(self, precios: np.ndarray) -> pd.DataFrame:
        return pl.optimizar_dia(precios, self.planta)

    def costo(self, plan: pd.DataFrame, precios: np.ndarray) -> float:
        return pl.costo_plan(plan, precios, self.planta)

    def validar(self, plan: pd.DataFrame) -> pd.DataFrame:
        return pl.validar_plan(plan, self.planta).rename(columns={"carga": "elemento"})

    def potencia_kw(self, plan: pd.DataFrame) -> pd.DataFrame:
        return pl.perfil_potencia(plan, self.planta)

    def horario(self, plan: pd.DataFrame) -> dict:
        return {c: _rangos([h for h in range(24) if plan.loc[h, c] == 1], 0) for c in self.elementos}

    def tabla_elementos(self) -> pd.DataFrame:
        return pd.DataFrame([{"Equipo": c.nombre, "kW": c.potencia_kw, "Horas/día": c.horas_dia,
                              "Corrida mínima (h)": c.horas_dia if c.continua else c.min_horas_seguidas,
                              "Horas permitidas": str(c.horas_permitidas), "Arranque actual": c.inicio_actual,
                              "Motivo": c.motivo_restriccion} for c in self.planta.cargas])


class PerfilFlota:
    tipo = "flota"
    etiqueta = ETIQUETAS["flota"]
    nombre_referencia = "Carga inmediata"
    unidad_elemento = "grupo"

    def __init__(self, flota: fl.Flota):
        self.flota = flota

    @property
    def nombre(self): return self.flota.nombre
    @property
    def descripcion(self): return self.flota.descripcion
    @property
    def fuentes(self): return self.flota.fuentes
    @property
    def supuestos(self): return self.flota.supuestos
    @property
    def hora_inicio(self): return self.flota.hora_inicio
    @property
    def exposicion_bolsa(self): return self.flota.exposicion_bolsa
    @exposicion_bolsa.setter
    def exposicion_bolsa(self, v): self.flota.exposicion_bolsa = v
    @property
    def limite_kw(self): return self.flota.potencia_patio_kw
    @limite_kw.setter
    def limite_kw(self, v): self.flota.potencia_patio_kw = v
    @property
    def elementos(self): return [g.nombre for g in self.flota.grupos]

    def errores(self) -> list[str]:
        return fl.validar_grupos(self.flota.grupos, self.hora_inicio)

    def plan_referencia(self) -> pd.DataFrame:
        return fl.carga_inmediata(self.flota.grupos, self.limite_kw, self.hora_inicio)

    def optimizar(self, precios: np.ndarray) -> pd.DataFrame | None:
        return fl.optimizar_carga(precios, self.flota.grupos, self.limite_kw, self.hora_inicio)

    def costo(self, plan: pd.DataFrame, precios: np.ndarray) -> float:
        return fl.costo_carga(plan, precios, self.exposicion_bolsa)

    def validar(self, plan: pd.DataFrame) -> pd.DataFrame:
        ancha = fl.verificar_plan(plan, self.flota.grupos, self.limite_kw, self.hora_inicio)
        reglas = {"Sale con la carga pedida": "R1 Sale con la carga pedida",
                  "Carga solo conectado": "R2 Carga solo mientras está conectado",
                  "Dentro de cargadores": "R3 Potencia ≤ cargadores del grupo",
                  "Dentro del límite del patio": "R4 Potencia ≤ límite del patio"}
        filas = []
        for _, f in ancha.iterrows():
            for col, regla in reglas.items():
                filas.append((regla, f["Grupo"], f[col] == "Sí"))
        t = pd.DataFrame(filas, columns=["regla", "elemento", "cumple"])
        t["resultado"] = np.where(t["cumple"], "✅ Cumple", "❌ No cumple")
        return t

    def potencia_kw(self, plan: pd.DataFrame) -> pd.DataFrame:
        return plan.copy()

    def horario(self, plan: pd.DataFrame) -> dict:
        return {g: _rangos([h for h in range(24) if plan.loc[h, g] > 0.5], self.hora_inicio)
                for g in self.elementos}

    def tabla_elementos(self) -> pd.DataFrame:
        return self.flota.tabla()


def cargar_perfil(tipo: str, ruta: str | Path | None = None):
    ruta = Path(ruta or CONFIG_POR_DEFECTO[tipo])
    if tipo == "fabrica":
        return PerfilFabrica(pl.Planta.desde_json(ruta))
    if tipo == "flota":
        return PerfilFlota(fl.Flota.desde_json(ruta))
    raise ValueError(f"Tipo de usuario desconocido: {tipo}")


# ---------------------------------------------------------------------------
# Funciones comunes a los dos perfiles
# ---------------------------------------------------------------------------
def etiquetas_horas(perfil) -> list[str]:
    return [f"{(perfil.hora_inicio + i) % 24:02d}:00" for i in range(24)]


def ventanas(serie: pd.Series, perfil, fechas=None) -> list:
    """Fechas cuya ventana de 24 h (desde perfil.hora_inicio) está completa en la serie."""
    candidatas = sorted(set(serie.index.date)) if fechas is None else list(fechas)
    return [f for f in candidatas if fl.precios_ventana(serie, f, perfil.hora_inicio) is not None]


def planear_periodo(pron: pd.Series, perfil, fechas=None) -> dict:
    """Plan óptimo de cada ventana del pronóstico y comparación con el plan de referencia."""
    ref = perfil.plan_referencia()
    dias = []
    for f in ventanas(pron, perfil, fechas):
        precios = fl.precios_ventana(pron, f, perfil.hora_inicio)
        opt = perfil.optimizar(precios)
        if opt is None:
            continue
        dias.append({"fecha": pd.Timestamp(f), "precios": precios, "plan_optimo": opt, "plan_referencia": ref,
                     "costo_referencia": perfil.costo(ref, precios), "costo_optimo": perfil.costo(opt, precios),
                     "validacion": perfil.validar(opt)})
    return {"dias": dias,
            "costo_referencia": sum(d["costo_referencia"] for d in dias),
            "costo_optimo": sum(d["costo_optimo"] for d in dias)}


def backtest(real: pd.Series, pred: pd.Series, perfil) -> pd.DataFrame:
    """Cada ventana del periodo de prueba: plan hecho con el PRONÓSTICO y pagado con el precio REAL.
    Compara contra el plan de referencia y contra el plan perfecto (precio conocido)."""
    ref = perfil.plan_referencia()
    filas = []
    for f in ventanas(real, perfil):
        pr = fl.precios_ventana(real, f, perfil.hora_inicio)
        pp = fl.precios_ventana(pred, f, perfil.hora_inicio)
        if pp is None:
            continue
        plan_p, plan_r = perfil.optimizar(pp), perfil.optimizar(pr)
        if plan_p is None or plan_r is None:
            continue
        filas.append({"fecha": pd.Timestamp(f), "costo_actual": perfil.costo(ref, pr),
                      "costo_con_pronostico": perfil.costo(plan_p, pr), "costo_perfecto": perfil.costo(plan_r, pr)})
    if not filas:
        return pd.DataFrame(columns=["costo_actual", "costo_con_pronostico", "costo_perfecto"])
    return pd.DataFrame(filas).set_index("fecha")


def ahorro_backtest(bt: pd.DataFrame) -> tuple[float, float]:
    """(ahorro real con pronóstico %, ahorro máximo con precio conocido %)."""
    if bt.empty or bt["costo_actual"].sum() == 0:
        return float("nan"), float("nan")
    base = bt["costo_actual"].sum()
    return (float(100 * (1 - bt["costo_con_pronostico"].sum() / base)),
            float(100 * (1 - bt["costo_perfecto"].sum() / base)))


def comparar_candidatos(ev: dict, bt_principal: pd.DataFrame, perfil) -> pd.DataFrame:
    """Todos los candidatos del modelo en el mismo periodo: error, horas baratas y ahorro real
    en backtest para ESTE perfil. Reutiliza el backtest del modelo principal."""
    real = ev["real"]
    base_total = bt_principal["costo_actual"].sum()
    filas = []
    for nombre, pred in ev["candidatos"].items():
        pred = pred.reindex(real.index)
        if nombre == NOMBRE_PRINCIPAL:
            total = bt_principal["costo_con_pronostico"].sum()
        else:
            total = 0.0
            for f in bt_principal.index:
                pr = fl.precios_ventana(real, f, perfil.hora_inicio)
                pp = fl.precios_ventana(pred, f, perfil.hora_inicio)
                plan = perfil.optimizar(pp) if pp is not None else None
                total += perfil.costo(plan if plan is not None else perfil.plan_referencia(), pr)
        filas.append({
            "Candidato": nombre,
            "MAE (COP/kWh)": float((pred - real).abs().mean()),
            "MAPE (%)": mape(real, pred),
            "Horas baratas acertadas (%)": acierto_horas_baratas(real, pred) * 100,
            "Ahorro real en backtest (%)": (1 - total / base_total) * 100 if base_total else float("nan"),
        })
    return pd.DataFrame(filas).sort_values("Ahorro real en backtest (%)", ascending=False).reset_index(drop=True)


def resumen_para_ia(plan: dict, pron: pd.Series, ev: dict, bt: pd.DataFrame, perfil, fuente: str) -> dict:
    """Solo números ya calculados: es lo único que recibe el LLM."""
    ahorro = plan["costo_referencia"] - plan["costo_optimo"]
    horario = perfil.horario(plan["dias"][0]["plan_optimo"]) if plan["dias"] else {}
    perfil_h = pron.groupby(pron.index.hour).mean()
    caras = [f"{h:02d}:00" for h in perfil_h.nlargest(4).index.sort_values()]
    baratas = [f"{h:02d}:00" for h in perfil_h.nsmallest(4).index.sort_values()]
    todas = pd.concat([d["validacion"] for d in plan["dias"]]) if plan["dias"] else pd.DataFrame({"cumple": []})
    m = ev["metricas"]
    ahorro_real, ahorro_max = ahorro_backtest(bt)
    return {
        "tipo_usuario": perfil.etiqueta.split(" ", 1)[1],
        "usuario": perfil.nombre, "fuente_datos": fuente,
        "plan_de_referencia": perfil.nombre_referencia,
        "ventanas_planeadas": len(plan["dias"]),
        "fechas": [f"{d['fecha']:%Y-%m-%d} desde las {perfil.hora_inicio:02d}:00" for d in plan["dias"]],
        "costo_referencia_cop": round(plan["costo_referencia"]),
        "costo_vatioplan_cop": round(plan["costo_optimo"]),
        "ahorro_cop": round(ahorro),
        "ahorro_pct": round(100 * ahorro / plan["costo_referencia"], 1) if plan["costo_referencia"] else 0,
        "horario_recomendado_primera_ventana": horario,
        "horas_mas_caras": caras, "horas_mas_baratas": baratas,
        "precio_max": round(float(pron.max())), "precio_min": round(float(pron.min())),
        "precio_promedio": round(float(pron.mean())),
        "mae_modelo": round(m["mae_modelo"]),
        "acierto_horas_baratas": round(m["acierto_horas_baratas_modelo"], 2),
        "ahorro_real_backtest_pct": round(ahorro_real, 1),
        "ahorro_maximo_backtest_pct": round(ahorro_max, 1),
        "dias_backtest": len(bt),
        "exposicion_bolsa": perfil.exposicion_bolsa,
        "limite_potencia_kw": perfil.limite_kw,
        "reglas": "todas cumplen" if todas["cumple"].all() else f"{int((~todas['cumple']).sum())} incumplimientos",
        "supuestos": perfil.supuestos,
    }
