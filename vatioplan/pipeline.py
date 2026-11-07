"""
pipeline.py: Une todos los pasos. Cada función es un paso que el orquestador
(Prefect, GitHub Actions o n8n) ejecuta en orden:

  1. datos      → descarga SIMEM / XM / NOAA (caché incremental en data/)
  2. modelo     → evalúa (periodo de prueba) y entrena el pronóstico
  3. pronóstico → precio hora a hora de los próximos días (común a todos los usuarios)
  4. planeación → por cada tipo de usuario (fábrica y flota): plan óptimo + reglas
  5. validación → backtest con precio real y comparación de modelos, por usuario
  6. informe    → IA generativa redacta un informe por usuario
  7. salidas    → archivos en salidas/<tipo>/ para el tablero y las notificaciones

Uso:  python -m vatioplan.pipeline                    (los dos usuarios)
      python -m vatioplan.pipeline --perfil flota     (solo uno)
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .datos import cargar_datos
from .informe import generar_informe
from .modelo import entrenar, evaluar, pronosticar
from .perfiles import (CONFIG_POR_DEFECTO, ahorro_backtest, backtest, cargar_perfil, comparar_candidatos,
                       planear_periodo, resumen_para_ia, ventanas)

RAIZ = Path(__file__).resolve().parent.parent
PERFILES = ["fabrica", "flota"]


def paso_datos(inicio="2023-01-01", fin=None, cache=None) -> pd.DataFrame:
    return cargar_datos(inicio, fin, carpeta_cache=cache or RAIZ / "data")


def paso_modelo(df: pd.DataFrame, dias_prueba: int = 90):
    ev = evaluar(df, dias_prueba)
    return entrenar(df), ev


def dias_a_planear(df: pd.DataFrame, pron: pd.Series, dias_plan: int = 3, hoy: date | None = None) -> list:
    """Fechas a planear: desde hoy (o el día siguiente al último dato) durante `dias_plan` días."""
    hoy = pd.Timestamp(hoy or date.today())
    if df.attrs.get("fuente") == "SIMULADO":            # en simulación "hoy" = último dato + 3 días
        hoy = df.index.max().normalize() + pd.Timedelta(days=3)
    inicio = max(hoy, df.index.max().normalize() + pd.Timedelta(days=1))
    fechas = [(inicio + pd.Timedelta(days=i)).date() for i in range(dias_plan)]
    disponibles = set(pron.index.date)
    fechas = [f for f in fechas if f in disponibles]
    if not fechas:                                      # si el retraso es mayor, los últimos días completos
        completos = pron.groupby(pron.index.date).size()
        fechas = [d for d, n in completos.items() if n == 24][-dias_plan:]
    return fechas


def paso_pronostico(df, modelo, dias_plan: int = 3, hoy: date | None = None) -> pd.Series:
    """Pronóstico de los días a planear (00:00 a 23:00). Se mantiene por compatibilidad con el notebook."""
    pron = pronosticar(df, modelo)
    fechas = dias_a_planear(df, pron, dias_plan, hoy)
    return pron[pd.Index(pron.index.date).isin(fechas)]


def paso_perfil(perfil, df, ev, pron_completo, fechas, comparar: bool = True) -> dict:
    """Planeación, validación, backtest, comparación de modelos y resumen para un usuario."""
    fuente = df.attrs.get("fuente", "REAL")
    plan = planear_periodo(pron_completo, perfil, ventanas(pron_completo, perfil, fechas))
    bt = backtest(ev["real"], ev["pred"], perfil)
    comparacion = comparar_candidatos(ev, bt, perfil) if comparar else None
    resumen = resumen_para_ia(plan, pron_completo, ev, bt, perfil, fuente)
    return {"perfil": perfil, "plan": plan, "backtest": bt, "comparacion": comparacion, "resumen": resumen}


def guardar_salidas(res: dict, carpeta: Path, texto: str, proveedor: str) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    perfil, plan, bt = res["perfil"], res["plan"], res["backtest"]
    bt.to_csv(carpeta / "backtest.csv")
    if res["comparacion"] is not None:
        res["comparacion"].to_csv(carpeta / "comparacion_modelos.csv", index=False)
    filas = []
    for d in plan["dias"]:
        for tipo in ["plan_referencia", "plan_optimo"]:
            p = perfil.potencia_kw(d[tipo]).copy()
            p.insert(0, "hora", [f"{(perfil.hora_inicio + i) % 24:02d}:00" for i in range(24)])
            p.insert(0, "tipo", tipo)
            p.insert(0, "ventana", f"{d['fecha']:%Y-%m-%d}")
            filas.append(p)
    if filas:
        pd.concat(filas).to_csv(carpeta / "planes_kw.csv", index=False)
        pd.concat([d["validacion"].assign(ventana=f"{d['fecha']:%Y-%m-%d}") for d in plan["dias"]]) \
            .to_csv(carpeta / "validacion.csv", index=False)
    resumen = dict(res["resumen"])
    if res["comparacion"] is not None:
        resumen["comparacion_modelos"] = res["comparacion"].round(1).to_dict("records")
    resumen["proveedor_ia"] = proveedor
    (carpeta / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
    (carpeta / "informe.md").write_text(texto, encoding="utf-8")


def ejecutar(perfiles: list[str] | None = None, rutas_config: dict | None = None, dias_plan: int = 3,
             inicio="2023-01-01", carpeta_salida: str | Path | None = None, verbose: bool = True,
             con_informe: bool = True) -> dict:
    """Corre todo el pipeline. Devuelve los datos y modelo comunes y un bloque por usuario."""
    perfiles = perfiles or PERFILES
    rutas_config = {**CONFIG_POR_DEFECTO, **(rutas_config or {})}
    salida = Path(carpeta_salida or RAIZ / "salidas")
    salida.mkdir(parents=True, exist_ok=True)

    df = paso_datos(inicio)
    fuente = df.attrs.get("fuente", "REAL")
    modelo, ev = paso_modelo(df)
    pron_completo = pronosticar(df, modelo)
    fechas = dias_a_planear(df, pron_completo, dias_plan)

    # --- salidas comunes ---
    df.tail(24 * 60).to_csv(salida / "historico_60d.csv")
    pron_completo.to_frame().to_csv(salida / "pronostico.csv")
    pd.DataFrame({"real": ev["real"], "modelo": ev["pred"], "base_ingenua": ev["base"]}).to_csv(salida / "evaluacion.csv")

    resultado = {"df": df, "fuente": fuente, "modelo": modelo, "evaluacion": ev, "pronostico": pron_completo,
                 "fechas": fechas, "perfiles": {}}
    for tipo in perfiles:
        perfil = cargar_perfil(tipo, rutas_config[tipo])
        res = paso_perfil(perfil, df, ev, pron_completo, fechas)
        texto, proveedor = generar_informe(res["resumen"]) if con_informe else ("", "sin informe")
        res["informe"], res["proveedor_ia"] = texto, proveedor
        guardar_salidas(res, salida / tipo, texto, proveedor)
        resultado["perfiles"][tipo] = res
        if verbose:
            real, maximo = ahorro_backtest(res["backtest"])
            print("\n" + "=" * 70)
            print(f"{perfil.etiqueta}: {perfil.nombre}")
            print(f"Ahorro real en backtest: {real:.1f}% (máximo {maximo:.1f}%)")
            print(texto)
            print(f"Informe generado con: {proveedor}, archivos en {salida / tipo}")

    estado = {"fuente": fuente, "ultimo_dato": f"{df.index.max():%Y-%m-%d %H:%M}",
              "fechas_planeadas": [f"{f}" for f in fechas], "metricas_modelo": ev["metricas"],
              "perfiles": {t: {"usuario": r["perfil"].nombre,
                               "ahorro_real_backtest_pct": r["resumen"]["ahorro_real_backtest_pct"],
                               "reglas": r["resumen"]["reglas"]} for t, r in resultado["perfiles"].items()}}
    (salida / "estado.json").write_text(json.dumps(estado, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return resultado


def incumplimientos(resultado: dict) -> dict:
    """Cantidad de reglas incumplidas por usuario (el orquestador no publica si hay alguna)."""
    salida = {}
    for tipo, res in resultado["perfiles"].items():
        dias = res["plan"]["dias"]
        salida[tipo] = int(sum((~d["validacion"]["cumple"]).sum() for d in dias))
    return salida


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline diario de VatioPlan")
    ap.add_argument("--perfil", choices=PERFILES, action="append",
                    help="Tipo de usuario a planear (se puede repetir). Por defecto, todos.")
    ap.add_argument("--dias", type=int, default=3, help="Días a planear")
    args = ap.parse_args()
    r = ejecutar(perfiles=args.perfil, dias_plan=args.dias)
    malos = {t: n for t, n in incumplimientos(r).items() if n}
    if malos:
        raise SystemExit(f"Reglas incumplidas: {malos}")
