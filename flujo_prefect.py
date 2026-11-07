"""
flujo_prefect.py: Orquestación con Prefect (gratis, open source).

Cada paso del pipeline es una tarea con reintentos. El flujo corre todos los días a las
6:15 a. m. hora Colombia, después de que XM publica sus datos (~4:00 a. m.), y planea los
dos tipos de usuario (fábrica y flota) con el mismo pronóstico.
Si un plan viola una regla de ingeniería, el flujo falla y NO publica ese informe.

Instalar:                pip install -r requirements-orquestacion.txt
Probar una vez:          python flujo_prefect.py --una-vez
Probar sin internet:     VATIOPLAN_PERMITIR_SIMULADO=1 python flujo_prefect.py --una-vez
Dejarlo programado:      python flujo_prefect.py          (deja un proceso escuchando)
Ver la interfaz gráfica: prefect server start             (http://127.0.0.1:4200)
"""
import os
import sys

from prefect import flow, get_run_logger, task

from vatioplan.informe import generar_informe
from vatioplan.modelo import pronosticar
from vatioplan.perfiles import CONFIG_POR_DEFECTO, cargar_perfil
from vatioplan.pipeline import RAIZ, PERFILES, dias_a_planear, guardar_salidas, paso_datos, paso_modelo, paso_perfil


@task(retries=3, retry_delay_seconds=600, name="1. Descargar datos (SIMEM, XM, NOAA)")
def t_datos():
    df = paso_datos()
    if df.attrs.get("fuente") != "REAL" and not os.getenv("VATIOPLAN_PERMITIR_SIMULADO"):
        raise RuntimeError("SIMEM no respondió: no se planea con datos simulados")
    return df


@task(name="2. Evaluar y entrenar modelo ML")
def t_modelo(df):
    return paso_modelo(df)


@task(name="3. Pronosticar precio")
def t_pronostico(df, modelo):
    pron = pronosticar(df, modelo)
    return pron, dias_a_planear(df, pron)


@task(name="4. Optimizar, verificar reglas y backtest")
def t_perfil(tipo, df, ev, pron, fechas):
    perfil = cargar_perfil(tipo, CONFIG_POR_DEFECTO[tipo])
    res = paso_perfil(perfil, df, ev, pron, fechas)
    incumple = sum(int((~d["validacion"]["cumple"]).sum()) for d in res["plan"]["dias"])
    if incumple:
        raise ValueError(f"{perfil.nombre}: el plan viola {incumple} reglas de ingeniería; no se publica")
    return res


@task(retries=2, retry_delay_seconds=60, name="5. Informe con IA generativa")
def t_informe(resumen):
    return generar_informe(resumen)


@task(name="6. Publicar salidas")
def t_publicar(tipo, res, texto, proveedor):
    guardar_salidas(res, RAIZ / "salidas" / tipo, texto, proveedor)
    get_run_logger().info("%s: ahorro esperado %s COP (%s%%), ahorro real en backtest %s%%",
                          res["perfil"].nombre, f"{res['resumen']['ahorro_cop']:,}",
                          res["resumen"]["ahorro_pct"], res["resumen"]["ahorro_real_backtest_pct"])
    # Aquí se puede agregar el envío por correo o Telegram (ver n8n_workflow.json)


@flow(name="VatioPlan diario", log_prints=True)
def vatioplan_diario(perfiles: list[str] = PERFILES):
    df = t_datos()
    modelo, ev = t_modelo(df)
    pron, fechas = t_pronostico(df, modelo)
    resumenes = {}
    for tipo in perfiles:
        res = t_perfil(tipo, df, ev, pron, fechas)
        texto, proveedor = t_informe(res["resumen"])
        t_publicar(tipo, res, texto, proveedor)
        resumenes[tipo] = res["resumen"]
    return resumenes


if __name__ == "__main__":
    if "--una-vez" in sys.argv:
        vatioplan_diario()
    else:
        from prefect.schedules import Cron
        vatioplan_diario.serve(name="vatioplan-diario", tags=["energia"],
                               schedules=[Cron("15 6 * * *", timezone="America/Bogota")])
