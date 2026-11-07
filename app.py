"""
app.py: Tablero de VatioPlan (Streamlit, gratis en Streamlit Community Cloud).

Dos tipos de usuario con el mismo pronóstico del precio de bolsa:
  🏭 Fábrica manufacturera  (caso de referencia: Cementos Argos, planta Rioclaro)
  🚌 Flota eléctrica         (caso de referencia: patio Fontibón I (Refugio), SITP Bogotá)

Ejecutar local:   streamlit run app.py
"""
import copy
import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vatioplan.flotas import grupos_desde_tabla, precios_ventana
from vatioplan.informe import generar_informe, plantilla
from vatioplan.perfiles import (ETIQUETAS, PerfilFlota, ahorro_backtest, backtest, cargar_perfil,
                                etiquetas_horas, planear_periodo, resumen_para_ia, ventanas)
from vatioplan.pipeline import ejecutar

st.set_page_config(page_title="VatioPlan", page_icon="⚡", layout="wide")
NOTA_PRONOSTICO = "Calculado con el precio pronosticado, no con el precio real."
COLORES = ["#1971c2", "#2f9e44", "#e67700", "#7048e8", "#d9480f", "#0c8599", "#c2255c"]


@st.cache_resource(show_spinner="Descargando datos, entrenando el modelo y planeando los dos usuarios…")
def correr_pipeline(_clave: int):
    return ejecutar(verbose=False, con_informe=False)


@st.cache_data(show_spinner="Calculando el ahorro real con precios reales (backtest)…")
def backtest_guardado(_perfil, _real, _pred, clave: str):
    """`clave` resume los parámetros del perfil: si cambian, se recalcula."""
    return backtest(_real, _pred, _perfil)


@st.cache_data(show_spinner="La IA está redactando el informe…")
def informe_ia(resumen_json: str):
    return generar_informe(json.loads(resumen_json))


def clave_perfil(perfil) -> str:
    if isinstance(perfil, PerfilFlota):
        datos = [vars(g) for g in perfil.flota.grupos]
    else:
        datos = [{k: v for k, v in vars(c).items() if k != "permitidas"} for c in perfil.planta.cargas]
    return json.dumps([perfil.tipo, perfil.limite_kw, perfil.hora_inicio, perfil.exposicion_bolsa, datos],
                      default=str)


def grafica_plan(titulo, kw: pd.DataFrame, precios, perfil):
    x = etiquetas_horas(perfil)
    fig = go.Figure()
    for i, col in enumerate(kw.columns):
        color = "#ced4da" if col == "Carga base" else COLORES[i % len(COLORES)]
        fig.add_bar(x=x, y=kw[col].values, name=col, marker_color=color)
    fig.add_scatter(x=x, y=precios, name="Precio pronosticado (COP/kWh)", yaxis="y2",
                    line=dict(color="black", width=2))
    fig.add_hline(y=perfil.limite_kw, line_dash="dash", annotation_text="Límite de potencia")
    fig.update_layout(barmode="stack", title=titulo, height=390, xaxis_title="Hora", yaxis_title="kW",
                      yaxis2=dict(overlaying="y", side="right", title="COP/kWh"),
                      legend=dict(orientation="h", y=-0.25), margin=dict(t=50))
    return fig


# ---------------- barra lateral ----------------
st.sidebar.title("⚡ VatioPlan")
st.sidebar.caption("Mueve el consumo flexible a las horas más baratas de la bolsa de energía.")
tipo = st.sidebar.radio("Tipo de usuario", ["fabrica", "flota"], format_func=lambda t: ETIQUETAS[t])
if st.sidebar.button("🔄 Actualizar datos y pronóstico"):
    st.session_state["clave"] = st.session_state.get("clave", 0) + 1
r = correr_pipeline(st.session_state.get("clave", 0))
ev, pron, m = r["evaluacion"], r["pronostico"], r["evaluacion"]["metricas"]

subido = st.sidebar.file_uploader("Usar una configuración propia (.json)", type="json",
                                  help="Mismo formato que los archivos de la carpeta config/. "
                                       "El campo \"tipo\" debe ser \"fabrica\" o \"flota\".")
perfil = copy.deepcopy(r["perfiles"][tipo]["perfil"])
if subido is not None:
    try:
        contenido = json.loads(subido.getvalue().decode("utf-8"))
        if contenido.get("tipo", tipo) != tipo:
            st.sidebar.warning(f"El archivo es de tipo «{contenido.get('tipo')}»; cambie el tipo de usuario.")
        else:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
                json.dump(contenido, tmp)
            perfil = cargar_perfil(tipo, tmp.name)
            Path(tmp.name).unlink(missing_ok=True)
    except Exception as e:  # el archivo no tiene el formato esperado
        st.sidebar.error(f"No se pudo leer la configuración: {e}")

st.sidebar.subheader("Parámetros")
perfil.exposicion_bolsa = st.sidebar.slider("Exposición a precio de bolsa", 0.0, 1.0, float(perfil.exposicion_bolsa),
                                            0.05, help="Fracción de la energía que se paga a precio de bolsa.")
etiqueta_limite = "Potencia máxima contratada (kW)" if tipo == "fabrica" else "Potencia máxima del patio (kW)"
perfil.limite_kw = st.sidebar.number_input(etiqueta_limite, 50, 100000, int(perfil.limite_kw), 100)
if tipo == "flota":
    perfil.flota.hora_inicio = st.sidebar.selectbox(
        "La ventana de 24 h empieza a las", list(range(24)), index=perfil.hora_inicio,
        format_func=lambda h: f"{h:02d}:00", help="Con 12:00, la carga nocturna queda en una sola ventana.")
st.sidebar.caption(f"Datos: {r['fuente']}, último precio publicado {r['df'].index.max():%d-%b-%Y %H:%M}")
if r["fuente"] == "SIMULADO":
    st.warning("⚠️ Sin conexión a SIMEM: se muestran DATOS SIMULADOS solo para probar la app.")

# ---------------- encabezado ----------------
st.title(f"{ETIQUETAS[tipo].split(' ', 1)[0]} {perfil.nombre}")
st.caption(perfil.descripcion)

# Para la flota, la tabla de vehículos es editable y se usa en todo el tablero
if tipo == "flota":
    with st.expander("🚌 Vehículos del patio (editable: cada fila es un grupo de vehículos iguales)", expanded=False):
        tabla = st.data_editor(perfil.flota.tabla(), num_rows="dynamic", hide_index=True, width="stretch",
                               key=f"tabla_flota_{subido.name if subido else 'base'}",
                               column_config={
                                   "Cantidad": st.column_config.NumberColumn(min_value=0, step=1),
                                   "Llega (hora)": st.column_config.NumberColumn(min_value=0, max_value=23, step=1),
                                   "Sale (hora)": st.column_config.NumberColumn(min_value=0, max_value=23, step=1),
                                   "Carga al llegar (%)": st.column_config.NumberColumn(min_value=0, max_value=100),
                                   "Carga al salir (%)": st.column_config.NumberColumn(min_value=0, max_value=100)})
    perfil.flota.grupos = grupos_desde_tabla(tabla)
    errores = perfil.errores() if perfil.flota.grupos else ["Agregue al menos un grupo de vehículos."]
    if errores:
        for e in errores:
            st.error(e)
        st.stop()

fechas = ventanas(pron, perfil, r["fechas"]) or ventanas(pron, perfil)
if not fechas:
    st.error("El pronóstico no cubre una ventana completa de 24 horas con esta hora de inicio.")
    st.stop()
fecha = st.sidebar.selectbox("Día a planear", fechas,
                             format_func=lambda d: f"{d:%a %d-%b-%Y} desde las {perfil.hora_inicio:02d}:00")
precios = precios_ventana(pron, fecha, perfil.hora_inicio)
ref, opt = perfil.plan_referencia(), perfil.optimizar(precios)
if opt is None:
    st.error("No hay un plan que cumpla todas las reglas con estos parámetros. Aumente el límite de potencia "
             "o el tiempo disponible.")
    st.stop()
validacion = perfil.validar(opt)
if not validacion["cumple"].all():
    st.warning("⚠️ Con estos parámetros no existe un plan que cumpla todas las reglas (por ejemplo, la potencia "
               "máxima no alcanza para las horas de producción). Se muestra el mejor plan aproximado; revise la "
               "tabla de reglas.")
c_ref, c_opt = perfil.costo(ref, precios), perfil.costo(opt, precios)
if clave_perfil(perfil) == clave_perfil(r["perfiles"][tipo]["perfil"]):
    bt = r["perfiles"][tipo]["backtest"]          # mismos parámetros que el pipeline: no se recalcula
else:
    bt = backtest_guardado(perfil, ev["real"], ev["pred"], clave_perfil(perfil))
ahorro_real, ahorro_max = ahorro_backtest(bt)

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Costo estimado con {perfil.nombre_referencia.lower()}", f"${c_ref:,.0f}", help=NOTA_PRONOSTICO)
k2.metric("Costo estimado con VatioPlan", f"${c_opt:,.0f}",
          f"-{(1 - c_opt / c_ref) * 100:.1f}%" if c_ref else None, delta_color="inverse", help=NOTA_PRONOSTICO)
k3.metric("Ahorro estimado del día (COP)", f"${c_ref - c_opt:,.0f}", help=NOTA_PRONOSTICO)
k4.metric("Ahorro real en backtest", f"{ahorro_real:.1f}%", f"máximo posible {ahorro_max:.1f}%",
          help=f"{len(bt)} días del periodo de prueba: cada uno se planea con el pronóstico y se paga con el "
               "precio real. Usa los parámetros actuales de la barra lateral.")
st.caption("Los costos del día usan el pronóstico. El ahorro real en backtest usa precios reales.")

tab_plan, tab_caso, tab_pron, tab_val, tab_inf = st.tabs(
    ["📋 Plan del día", "🧾 Caso real y supuestos", "📈 Pronóstico", "🧪 Validación del modelo", "🤖 Informe IA"])

with tab_plan:
    for titulo, plan in [(perfil.nombre_referencia, ref), ("Plan VatioPlan (optimizado)", opt)]:
        st.plotly_chart(grafica_plan(titulo, perfil.potencia_kw(plan), precios, perfil), width="stretch")
    horario = pd.DataFrame({perfil.unidad_elemento.capitalize(): perfil.elementos,
                            "VatioPlan": list(perfil.horario(opt).values()),
                            perfil.nombre_referencia: list(perfil.horario(ref).values())})
    st.subheader("Horario recomendado")
    st.dataframe(horario, hide_index=True, width="stretch")
    if tipo == "flota":
        faltante = sum(g.energia_red_kwh() for g in perfil.flota.grupos) - ref.values.sum()
        if faltante > 0.5:
            st.warning(f"Con carga inmediata faltarían {faltante:,.0f} kWh porque el patio se queda corto en las "
                       "horas de llegada. VatioPlan sí completa la carga.")
    st.subheader("Reglas de ingeniería verificadas")
    st.dataframe(validacion[["regla", "elemento", "resultado"]], hide_index=True, width="stretch")

with tab_caso:
    st.markdown(f"**{perfil.nombre}**\n\n{perfil.descripcion}")
    st.dataframe(perfil.tabla_elementos(), hide_index=True, width="stretch")
    if perfil.supuestos:
        st.info(f"**Supuestos:** {perfil.supuestos}")
    if perfil.fuentes:
        st.markdown("**Fuentes**\n" + "\n".join(f"- {f}" for f in perfil.fuentes))
    st.caption("Para otro cliente: copie el archivo de config/, cambie los valores y súbalo en la barra lateral.")

with tab_pron:
    hist = r["df"]["precio"].tail(24 * 21)
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, name="Real (SIMEM)", line=dict(color="#495057"))
    fig.add_scatter(x=pron.index, y=pron.values, name="Pronóstico", line=dict(color="#d9480f", dash="dot"))
    fig.update_layout(height=420, yaxis_title="COP/kWh", title="Precio de bolsa: últimas 3 semanas + pronóstico")
    st.plotly_chart(fig, width="stretch")
    st.caption("El precio real se publica con ~3 días de retraso; el pronóstico cubre ese hueco y los días a planear. "
               "Es el mismo pronóstico para los dos tipos de usuario.")

with tab_val:
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE modelo", f"{m['mae_modelo']:.0f} COP/kWh", f"{-m['mejora_mae_pct']:.0f}% vs. ingenuo",
              delta_color="inverse")
    c2.metric("Horas baratas acertadas", f"{m['acierto_horas_baratas_modelo']:.0%}",
              f"{(m['acierto_horas_baratas_modelo'] - m['acierto_horas_baratas_base']) * 100:+.0f} pts vs. ingenuo")
    c3.metric("Ahorro real en backtest", f"{ahorro_real:.1f}%", f"máximo posible {ahorro_max:.1f}%")
    st.subheader("Comparación de modelos en el periodo de prueba")
    comparacion = r["perfiles"][tipo]["comparacion"]
    st.caption(f"Periodo de prueba: {m['periodo_prueba']}. El ahorro de cada candidato se mide para este tipo "
               "de usuario con los parámetros del archivo de configuración.")
    if comparacion is not None:
        t = comparacion.round(1)
        t["MAE (COP/kWh)"] = t["MAE (COP/kWh)"].round(0).astype(int)
        st.dataframe(t, hide_index=True, width="stretch")
    st.caption("Modelo principal: gradient boosting que corrige el promedio de la misma hora en las últimas 4 semanas "
               "según la hora, el día de la semana y los festivos. Las otras filas quedan como referencia.")
    ult = pd.DataFrame({"Real": ev["real"], "Modelo ML (GB sobre promedio 4 semanas)": ev["pred"],
                        "Ingenuo (hace 7 días)": ev["base"]}).tail(24 * 14)
    st.plotly_chart(px.line(ult, labels={"value": "COP/kWh", "index": ""},
                            title="Últimas 2 semanas del periodo de prueba"), width="stretch")
    if len(bt):
        acumulado = (bt["costo_actual"] - bt["costo_con_pronostico"]).cumsum()
        st.plotly_chart(px.area(acumulado, labels={"value": "COP", "fecha": ""},
                                title="Ahorro acumulado si se hubiera seguido a VatioPlan (precios reales)"),
                        width="stretch")

with tab_inf:
    plan_periodo = planear_periodo(pron, perfil, fechas)
    resumen = resumen_para_ia(plan_periodo, pron, ev, bt, perfil, r["fuente"])
    st.caption("El informe usa los parámetros actuales de la barra lateral. La IA solo redacta: las cifras vienen "
               "del pipeline.")
    if st.button("✍️ Redactar con IA (Gemini / Groq)"):
        st.session_state[f"ia_{tipo}"] = json.dumps(resumen, ensure_ascii=False, default=str)
    if st.session_state.get(f"ia_{tipo}") == json.dumps(resumen, ensure_ascii=False, default=str):
        texto, proveedor = informe_ia(st.session_state[f"ia_{tipo}"])
    else:
        texto, proveedor = plantilla(resumen), "plantilla (pulse el botón para redactar con IA)"
    st.markdown(texto)
    st.caption(f"Redactado con: {proveedor}.")
    st.download_button("⬇️ Descargar informe", texto, file_name=f"informe_vatioplan_{tipo}.md")
    with st.expander("Datos que recibe la IA (JSON)"):
        st.json(resumen)
