"""
pipeline.py — Une todos los pasos. Cada función es un paso que el orquestador
(Prefect o n8n) ejecuta en orden:

  1. datos      → descarga SIMEM / XM / NOAA
  2. modelo     → evalúa (backtest) y entrena el pronóstico
  3. pronóstico → precio hora a hora de los próximos días
  4. planeación → programa óptimo de cargas + validación de reglas
  5. informe    → IA generativa redacta el informe
  6. salidas    → archivos en /salidas para el tablero y las notificaciones

Uso:  python -m vatioplan.pipeline            (o  python pipeline.py  desde la raíz)
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from .datos import cargar_datos
from .informe import generar_informe
from .modelo import entrenar, evaluar, pronosticar
from .planta import Planta, backtest, planear_periodo, validar_plan

RAIZ = Path(__file__).resolve().parent.parent


def rangos_horas(horas: list[int]) -> str:
    """[2,3,4,13] → '02:00–05:00, 13:00–14:00'"""
    if not horas:
        return "no opera"
    horas, bloques, ini = sorted(horas), [], None
    for i, h in enumerate(horas):
        ini = h if ini is None else ini
        if i == len(horas) - 1 or horas[i + 1] != h + 1:
            bloques.append(f"{ini:02d}:00–{(h + 1) % 24:02d}:00")
            ini = None
    return ", ".join(bloques)


def paso_datos(inicio="2023-01-01", fin=None, cache=None) -> pd.DataFrame:
    return cargar_datos(inicio, fin, carpeta_cache=cache or RAIZ / "data")


def paso_modelo(df: pd.DataFrame, dias_prueba: int = 90):
    ev = evaluar(df, dias_prueba)
    return entrenar(df), ev


def paso_pronostico(df, modelo, dias_plan: int = 3, hoy: date | None = None) -> pd.Series:
    pron = pronosticar(df, modelo)
    hoy = pd.Timestamp(hoy or date.today())
    if df.attrs.get("fuente") == "SIMULADO":            # en simulación "hoy" = último dato + 3 días
        hoy = df.index.max().normalize() + pd.Timedelta(days=3)
    inicio = max(hoy, df.index.max().normalize() + pd.Timedelta(days=1))
    fin = inicio + pd.Timedelta(days=dias_plan)
    sel = pron[(pron.index >= inicio) & (pron.index < fin)]
    if sel.empty:  # si el retraso es mayor, planea los días completos disponibles
        dias_completos = pron.groupby(pron.index.date).size()
        dias_ok = [pd.Timestamp(d) for d, n in dias_completos.items() if n == 24][-dias_plan:]
        sel = pron[pron.index.normalize().isin(dias_ok)]
    return sel


def paso_planeacion(pron: pd.Series, planta: Planta) -> dict:
    return planear_periodo(pron, planta)


def resumen_para_ia(plan: dict, pron: pd.Series, ev: dict, planta: Planta, fuente: str) -> dict:
    ahorro = plan["costo_actual"] - plan["costo_optimo"]
    dia0 = plan["dias"][0]["plan_optimo"]
    horario = {}
    for c in planta.cargas:
        horas = [h for h in range(24) if dia0.loc[h, c.nombre] == 1]
        horario[c.nombre] = rangos_horas(horas)
    perfil = pron.groupby(pron.index.hour).mean()
    caras = [f"{h:02d}:00" for h in perfil.nlargest(4).index.sort_values()]
    todas = pd.concat([d["validacion"] for d in plan["dias"]])
    m = ev["metricas"]
    return {
        "planta": planta.nombre, "fuente": fuente,
        "dias": len(plan["dias"]),
        "fechas": [f"{d['fecha']:%Y-%m-%d}" for d in plan["dias"]],
        "costo_actual_cop": round(plan["costo_actual"]),
        "costo_optimo_cop": round(plan["costo_optimo"]),
        "ahorro_cop": round(ahorro),
        "ahorro_pct": round(100 * ahorro / plan["costo_actual"], 1) if plan["costo_actual"] else 0,
        "horario_recomendado": horario,
        "horas_mas_caras": caras,
        "precio_max": round(float(pron.max())), "precio_min": round(float(pron.min())),
        "precio_promedio": round(float(pron.mean())),
        "mae_modelo": round(m["mae_modelo"]), "mae_base": round(m["mae_base"]),
        "acierto_horas_baratas": round(m["acierto_horas_baratas_modelo"], 2),
        "exposicion_bolsa": planta.exposicion_bolsa,
        "reglas": "todas cumplen" if todas["cumple"].all()
                  else f"{(~todas['cumple']).sum()} incumplimientos",
    }


def ejecutar(ruta_planta: str | Path | None = None, dias_plan: int = 3, inicio="2023-01-01",
             carpeta_salida: str | Path | None = None, verbose: bool = True) -> dict:
    salida = Path(carpeta_salida or RAIZ / "salidas")
    salida.mkdir(parents=True, exist_ok=True)
    planta = Planta.desde_json(ruta_planta or RAIZ / "config" / "planta_ejemplo.json")

    df = paso_datos(inicio)
    fuente = df.attrs.get("fuente", "REAL")
    modelo, ev = paso_modelo(df)
    pron = paso_pronostico(df, modelo, dias_plan)
    plan = paso_planeacion(pron, planta)
    bt = backtest(ev["real"], ev["pred"], planta)
    resumen = resumen_para_ia(plan, pron, ev, planta, fuente)
    texto, proveedor = generar_informe(resumen)

    # --- salidas para el tablero / notificaciones ---
    df.tail(24 * 60).to_csv(salida / "historico_60d.csv")
    pron.to_frame().to_csv(salida / "pronostico.csv")
    pd.DataFrame({"real": ev["real"], "modelo": ev["pred"], "base_ingenua": ev["base"]}).to_csv(salida / "evaluacion.csv")
    bt.to_csv(salida / "backtest.csv")
    filas = []
    for d in plan["dias"]:
        for tipo in ["plan_actual", "plan_optimo"]:
            p = d[tipo].copy()
            p.insert(0, "hora", range(24))
            p.insert(0, "tipo", tipo)
            p.insert(0, "fecha", f"{d['fecha']:%Y-%m-%d}")
            filas.append(p)
    pd.concat(filas).to_csv(salida / "planes.csv", index=False)
    pd.concat([d["validacion"].assign(fecha=f"{d['fecha']:%Y-%m-%d}") for d in plan["dias"]]) \
        .to_csv(salida / "validacion.csv", index=False)
    resumen["metricas_modelo"] = ev["metricas"]
    resumen["backtest"] = {
        "dias": len(bt),
        "ahorro_real_con_pronostico_pct": round(float(100 * (1 - bt["costo_con_pronostico"].sum() / bt["costo_actual"].sum())), 1),
        "ahorro_maximo_teorico_pct": round(float(100 * (1 - bt["costo_perfecto"].sum() / bt["costo_actual"].sum())), 1),
    }
    resumen["proveedor_ia"] = proveedor
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (salida / "informe.md").write_text(texto, encoding="utf-8")

    if verbose:
        print("\n" + "=" * 70)
        print(texto)
        print("=" * 70)
        print(f"Informe generado con: {proveedor} · archivos en {salida}")
    return {"df": df, "modelo": modelo, "evaluacion": ev, "pronostico": pron, "plan": plan,
            "backtest": bt, "resumen": resumen, "informe": texto, "planta": planta}


if __name__ == "__main__":
    ejecutar()
