"""
app.py — Tablero de VatioPlan (Streamlit, gratis en Streamlit Community Cloud).

Ejecutar local:   streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vatioplan.pipeline import ejecutar
from vatioplan.planta import Planta, costo_plan, optimizar_dia, perfil_potencia, plan_actual, validar_plan

st.set_page_config(page_title="VatioPlan", page_icon="⚡", layout="wide")
RAIZ = Path(__file__).parent
SALIDAS = RAIZ / "salidas"


@st.cache_resource(show_spinner="Descargando datos, entrenando el modelo y planeando…")
def correr_pipeline(_clave: int):
    return ejecutar(verbose=False)


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

tab1, tab2, tab3, tab4 = st.tabs(["📋 Plan del día", "📈 Pronóstico", "🧪 Validación del modelo", "🤖 Informe IA"])

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
    ult = pd.DataFrame({"Real": ev["real"], "Modelo ML": ev["pred"], "Ingenuo (hace 7 días)": ev["base"]}).tail(24 * 14)
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
