"""
app.py — Tablero de VatioPlan (Streamlit, gratis en Streamlit Community Cloud).

Ejecutar local:   streamlit run app.py
"""
import json
from dataclasses import astuple
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vatioplan.flotas import (GrupoVehiculos, backtest_flota, carga_inmediata, costo_carga, grupos_desde_tabla,
                               grupos_ejemplo, optimizar_carga, precios_ventana, rangos_ventana,
                               validar_grupos, ventanas_disponibles, verificar_plan)
from vatioplan.pipeline import ejecutar
from vatioplan.planta import Planta, costo_plan, optimizar_dia, perfil_potencia, plan_actual, validar_plan

st.set_page_config(page_title="VatioPlan", page_icon="⚡", layout="wide")
RAIZ = Path(__file__).parent
SALIDAS = RAIZ / "salidas"


@st.cache_resource(show_spinner="Descargando datos, entrenando el modelo y planeando…")
def correr_pipeline(_clave: int):
    return ejecutar(verbose=False)


@st.cache_data(show_spinner="Calculando el ahorro real de la flota en el periodo de prueba…")
def backtest_flota_guardado(_real, _pred, clave_pipeline: int, grupos_tupla: tuple, potencia_patio: float,
                            hora_inicio: int):
    grupos = [GrupoVehiculos(*fila) for fila in grupos_tupla]
    return backtest_flota(_real, _pred, grupos, potencia_patio, hora_inicio)


# ---------------- barra lateral ----------------
st.sidebar.title("⚡ VatioPlan")
st.sidebar.caption("Programa la producción en las horas más baratas de la energía.")
if st.sidebar.button("🔄 Actualizar datos y pronóstico"):
    st.session_state["clave"] = st.session_state.get("clave", 0) + 1
r = correr_pipeline(st.session_state.get("clave", 0))

planta: Planta = r["planta"]
st.sidebar.subheader("Parámetros de la planta")
planta.exposicion_bolsa = st.sidebar.slider("Exposición a precio de bolsa", 0.0, 1.0, planta.exposicion_bolsa, 0.05,
                                            help="Fracción de la energía que la empresa paga a precio de bolsa.")
planta.potencia_maxima_kw = st.sidebar.number_input("Potencia máxima contratada (kW)", 500, 2000,
                                                    int(planta.potencia_maxima_kw), 25)
fuente = r["df"].attrs.get("fuente", "REAL")
if fuente == "SIMULADO":
    st.warning("⚠️ Sin conexión a SIMEM: se muestran DATOS SIMULADOS solo para probar la app.")

# re-planea con los parámetros de la barra lateral
pron = r["pronostico"]
dias = sorted(set(pron.index.date))
dia = st.sidebar.selectbox("Día a planear", dias, format_func=lambda d: d.strftime("%a %d-%b-%Y"))
precios = pron[pron.index.date == dia].values
opt, act = optimizar_dia(precios, planta), plan_actual(planta)
c_act, c_opt = costo_plan(act, precios, planta), costo_plan(opt, precios, planta)

# ---------------- encabezado ----------------
st.title(f"Plan de energía · {planta.nombre}")
k1, k2, k3, k4 = st.columns(4)
nota_pronostico = "Calculado con el precio pronosticado, no con el precio real."
k1.metric("Costo estimado con operación actual", f"${c_act:,.0f}", help=nota_pronostico)
k2.metric("Costo estimado con VatioPlan", f"${c_opt:,.0f}", f"-{(1 - c_opt / c_act) * 100:.1f}%",
          delta_color="inverse", help=nota_pronostico)
k3.metric("Ahorro estimado del día (COP)", f"${c_act - c_opt:,.0f}", help=nota_pronostico)
m = r["evaluacion"]["metricas"]
k4.metric("Horas baratas acertadas", f"{m['acierto_horas_baratas_modelo']:.0%}",
          f"{(m['acierto_horas_baratas_modelo'] - m['acierto_horas_baratas_base']) * 100:+.0f} pts vs. ingenuo")
st.caption("Los costos y el ahorro del día usan el pronóstico. El ahorro medido con precios reales "
           "está en la pestaña Validación del modelo (ahorro real en backtest).")

tab1, tab_flotas, tab2, tab3, tab4 = st.tabs(["📋 Plan del día", "🚌 Flotas eléctricas", "📈 Pronóstico",
                                              "🧪 Validación del modelo", "🤖 Informe IA"])

with tab1:
    colores = ["#9aa5b1", "#d9480f", "#1971c2", "#2f9e44", "#7048e8", "#e67700"]
    for titulo, plan in [("Operación actual", act), ("Plan VatioPlan (optimizado)", opt)]:
        perf = perfil_potencia(plan, planta)
        fig = go.Figure()
        for i, col in enumerate(perf.columns):
            fig.add_bar(x=perf.index, y=perf[col], name=col, marker_color=colores[i % len(colores)])
        fig.add_scatter(x=list(range(24)), y=precios, name="Precio (COP/kWh)", yaxis="y2",
                        line=dict(color="black", width=2))
        fig.add_hline(y=planta.potencia_maxima_kw, line_dash="dash", annotation_text="Potencia máxima")
        fig.update_layout(barmode="stack", title=titulo, height=380, xaxis_title="Hora",
                          yaxis_title="kW", yaxis2=dict(overlaying="y", side="right", title="COP/kWh"),
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    st.subheader("Reglas de ingeniería")
    st.dataframe(validar_plan(opt, planta)[["regla", "carga", "resultado"]], hide_index=True, width="stretch")

with tab_flotas:
    st.subheader("Carga inteligente de flotas eléctricas")
    st.caption("Programa la carga de buses, furgonetas o montacargas eléctricos en las horas más baratas "
               "y garantiza que cada vehículo salga con la carga que necesita.")
    col_a, col_b, col_c = st.columns(3)
    potencia_patio = col_a.number_input("Potencia máxima del patio (kW)", 50, 20000, 1000, 50, key="flota_patio")
    hora_inicio = col_b.selectbox("La ventana de 24 horas empieza a las", list(range(24)), index=12,
                                  format_func=lambda h: f"{h:02d}:00", key="flota_inicio",
                                  help="Con 12:00, una carga nocturna de 21:00 a 05:00 queda dentro de una sola ventana.")
    exposicion_flota = col_c.slider("Exposición a precio de bolsa de la flota", 0.0, 1.0, 1.0, 0.05,
                                    key="flota_exposicion",
                                    help="Fracción de la energía de carga que se paga a precio de bolsa.")
    st.markdown("**Vehículos.** Edite la tabla o agregue filas. Cada fila es un grupo de vehículos iguales.")
    tabla_flota = st.data_editor(
        grupos_ejemplo(), num_rows="dynamic", hide_index=True, width="stretch", key="flota_tabla",
        column_config={
            "Cantidad": st.column_config.NumberColumn(min_value=0, step=1),
            "Batería (kWh)": st.column_config.NumberColumn(min_value=1),
            "Cargador (kW)": st.column_config.NumberColumn(min_value=1),
            "Llega (hora)": st.column_config.NumberColumn(min_value=0, max_value=23, step=1),
            "Sale (hora)": st.column_config.NumberColumn(min_value=0, max_value=23, step=1),
            "Carga al llegar (%)": st.column_config.NumberColumn(min_value=0, max_value=100),
            "Carga al salir (%)": st.column_config.NumberColumn(min_value=0, max_value=100),
        })
    grupos = grupos_desde_tabla(tabla_flota)
    errores_flota = validar_grupos(grupos, hora_inicio)
    fechas_flota = ventanas_disponibles(pron, hora_inicio)

    if not grupos:
        st.info("Agregue al menos un grupo de vehículos en la tabla para calcular el plan.")
    elif errores_flota:
        for error in errores_flota:
            st.error(error)
    elif not fechas_flota:
        st.info(f"El pronóstico no cubre una ventana completa de 24 horas que empiece a las {hora_inicio:02d}:00. "
                "Pruebe otra hora de inicio.")
    else:
        fecha_flota = st.selectbox("Ventana a planear", fechas_flota, key="flota_fecha",
                                   format_func=lambda d: f"Desde el {d:%a %d-%b-%Y} a las {hora_inicio:02d}:00")
        precios_flota = precios_ventana(pron, fecha_flota, hora_inicio)
        plan_inmediato = carga_inmediata(grupos, potencia_patio, hora_inicio)
        plan_flota = optimizar_carga(precios_flota, grupos, potencia_patio, hora_inicio)

        if plan_flota is None:
            st.error(f"No hay forma de cargar toda la flota con {potencia_patio:,} kW en el patio. "
                     "Aumente la potencia del patio, el tiempo conectado o baje la carga al salir.")
        else:
            energia_total = sum(g.energia_red_kwh() for g in grupos)
            faltante_inmediata = energia_total - plan_inmediato.values.sum()
            if faltante_inmediata > 0.5:
                st.warning(f"Con carga inmediata faltarían {faltante_inmediata:,.0f} kWh porque el patio se queda "
                           "corto en las horas de llegada. VatioPlan sí completa la carga.")

            c_inm = costo_carga(plan_inmediato, precios_flota, exposicion_flota)
            c_flota = costo_carga(plan_flota, precios_flota, exposicion_flota)
            grupos_tupla = tuple(astuple(g) for g in grupos)
            bt_flota = backtest_flota_guardado(r["evaluacion"]["real"], r["evaluacion"]["pred"],
                                               st.session_state.get("clave", 0), grupos_tupla,
                                               float(potencia_patio), int(hora_inicio))

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Costo estimado con carga inmediata", f"${c_inm:,.0f}", help=nota_pronostico)
            f2.metric("Costo estimado con VatioPlan", f"${c_flota:,.0f}",
                      f"-{(1 - c_flota / c_inm) * 100:.1f}%" if c_inm > 0 else None,
                      delta_color="inverse", help=nota_pronostico)
            f3.metric("Ahorro estimado de la noche (COP)", f"${c_inm - c_flota:,.0f}", help=nota_pronostico)
            if len(bt_flota) > 0:
                total_bt = bt_flota.sum()
                f4.metric("Ahorro real en backtest",
                          f"{(1 - total_bt['costo_con_pronostico'] / total_bt['costo_inmediata']) * 100:.1f}%",
                          f"máximo posible {(1 - total_bt['costo_perfecto'] / total_bt['costo_inmediata']) * 100:.1f}%",
                          help=f"{len(bt_flota)} ventanas del periodo de prueba: cada una se planea con el "
                               "pronóstico del modelo principal y se paga con el precio real.")

            etiquetas_horas = [f"{(hora_inicio + i) % 24:02d}:00" for i in range(24)]
            colores_flota = ["#1971c2", "#2f9e44", "#e67700", "#7048e8", "#d9480f", "#0c8599"]
            for titulo, plan in [("Carga inmediata (operación actual)", plan_inmediato),
                                 ("Plan VatioPlan (optimizado)", plan_flota)]:
                fig = go.Figure()
                for i, grupo in enumerate(plan.columns):
                    fig.add_bar(x=etiquetas_horas, y=plan[grupo], name=grupo,
                                marker_color=colores_flota[i % len(colores_flota)])
                fig.add_scatter(x=etiquetas_horas, y=precios_flota, name="Precio (COP/kWh)", yaxis="y2",
                                line=dict(color="black", width=2))
                fig.add_hline(y=potencia_patio, line_dash="dash", annotation_text="Potencia máxima del patio")
                fig.update_layout(barmode="stack", title=titulo, height=380, xaxis_title="Hora",
                                  yaxis_title="kW", yaxis2=dict(overlaying="y", side="right", title="COP/kWh"),
                                  legend=dict(orientation="h", y=-0.25))
                st.plotly_chart(fig, width="stretch")

            st.subheader("Horario de carga")
            horario = pd.DataFrame({
                "Grupo": [g.nombre for g in grupos],
                "VatioPlan": [rangos_ventana(plan_flota[g.nombre], hora_inicio) for g in grupos],
                "Carga inmediata": [rangos_ventana(plan_inmediato[g.nombre], hora_inicio) for g in grupos],
            })
            st.dataframe(horario, hide_index=True, width="stretch")
            st.subheader("Reglas verificadas")
            st.dataframe(verificar_plan(plan_flota, grupos, potencia_patio, hora_inicio), hide_index=True,
                         width="stretch")
            st.caption("Los costos de la noche usan el precio pronosticado. La carga se reparte por igual entre "
                       "los vehículos de cada grupo. Se asume una eficiencia de carga de 90%.")

with tab2:
    hist = r["df"]["precio"].tail(24 * 21)
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, name="Real (SIMEM)", line=dict(color="#495057"))
    fig.add_scatter(x=pron.index, y=pron.values, name="Pronóstico", line=dict(color="#d9480f", dash="dot"))
    fig.update_layout(height=420, yaxis_title="COP/kWh", title="Precio de bolsa: últimas 3 semanas + pronóstico")
    st.plotly_chart(fig, width="stretch")
    st.caption("El precio real se publica con ~3 días de retraso; por eso el pronóstico cubre ese hueco y los días a planear.")

with tab3:
    ev = r["evaluacion"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE modelo", f"{m['mae_modelo']:.0f} COP/kWh", f"{-m['mejora_mae_pct']:.0f}% vs. ingenuo", delta_color="inverse")
    c2.metric("MAPE modelo", f"{m['mape_modelo']:.1f}%")
    bt = r["backtest"]
    c3.metric("Ahorro real en backtest", f"{(1 - bt['costo_con_pronostico'].sum() / bt['costo_actual'].sum()) * 100:.1f}%",
              f"máximo posible {(1 - bt['costo_perfecto'].sum() / bt['costo_actual'].sum()) * 100:.1f}%")
    st.subheader("Comparación de modelos en el periodo de prueba")
    st.caption(f"Periodo de prueba: {m['periodo_prueba']}. Todos los candidatos se evalúan en los mismos días. "
               "El ahorro real se mide planeando con cada pronóstico y pagando con el precio real.")
    tabla_comparacion = r["comparacion"].round(1)
    tabla_comparacion["MAE (COP/kWh)"] = tabla_comparacion["MAE (COP/kWh)"].round(0).astype(int)
    st.dataframe(tabla_comparacion, hide_index=True, width="stretch")
    st.caption("Modelo principal: gradient boosting que corrige el promedio del precio de la misma hora en las "
               "últimas 4 semanas, según la hora, el día de la semana y los festivos. Las versiones anteriores "
               "del gradient boosting quedan en la tabla como referencia.")
    ult = pd.DataFrame({"Real": ev["real"], "Modelo ML (GB sobre promedio 4 semanas)": ev["pred"],
                        "Ingenuo (hace 7 días)": ev["base"]}).tail(24 * 14)
    st.plotly_chart(px.line(ult, labels={"value": "COP/kWh", "index": ""},
                            title="Últimas 2 semanas del periodo de prueba"), width="stretch")
    ahorro = (bt["costo_actual"] - bt["costo_con_pronostico"]).cumsum()
    st.plotly_chart(px.area(ahorro, labels={"value": "COP", "fecha": ""},
                            title="Ahorro acumulado si la planta hubiera seguido a VatioPlan (precios reales)"),
                    width="stretch")

with tab4:
    st.markdown(r["informe"])
    st.caption(f"Redactado con: {r['resumen'].get('proveedor_ia', '—')}. Las cifras vienen del pipeline; la IA solo redacta.")
    st.download_button("⬇️ Descargar informe", r["informe"], file_name="informe_vatioplan.md")
