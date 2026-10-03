# ⚡ VatioPlan

**Programar la producción industrial según el precio de la energía en Colombia.**
Proyecto final — Herramientas de Inteligencia Artificial, Universidad de los Andes (2026).

VatioPlan pronostica el **precio de bolsa horario** de la energía con machine learning, reprograma las
**cargas flexibles** de una planta a las horas más baratas respetando las **reglas de ingeniería**, y
entrega al jefe de planta un **informe redactado con IA generativa**. Todo corre solo cada mañana.

```
SIMEM/XM + NOAA ─▶ Datos ─▶ Modelo ML ─▶ Pronóstico ─▶ Optimización + reglas ─▶ IA generativa ─▶ Informe / Tablero
                                     ▲ orquestado todos los días 6:00 a. m. (Prefect o n8n) ▲
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `vatioplan/datos.py` | Descarga automática: precio horario (SIMEM `EC6945`), embalses y aportes (API XM), ONI (NOAA). Caché incremental en `data/`. |
| `vatioplan/features.py` | Variables del modelo, todas rezagadas ≥ 7 días (sin fuga de datos). |
| `vatioplan/modelo.py` | Gradient boosting (scikit-learn), comparación contra método ingenuo, métrica de "horas baratas acertadas". |
| `vatioplan/planta.py` | Optimización (PuLP), reglas R1–R4, validación "Cumple / No cumple", backtest de ahorro. |
| `vatioplan/informe.py` | Informe con Gemini o Groq (gratis); plantilla si no hay API key. |
| `vatioplan/pipeline.py` | Une todos los pasos y escribe `salidas/`. |
| `config/planta_ejemplo.json` | La planta: equipos, potencias, horarios permitidos. **Edítenlo para otra planta.** |
| `notebooks/VatioPlan_completo.ipynb` | Recorrido completo, paso a paso, con gráficas. |
| `app.py` | Tablero Streamlit (low-code). |
| `flujo_prefect.py` | Orquestación con Prefect (programado 6:00 a. m. Bogotá). |
| `n8n_workflow.json` | Orquestación visual alternativa en n8n (envía el informe por correo). |

## Cómo correrlo

### Opción A — Google Colab (lo más fácil)
1. Abran `notebooks/VatioPlan_completo.ipynb` en https://colab.research.google.com (Archivo → Subir notebook).
2. Ejecuten la primera celda y suban `vatioplan.zip` cuando lo pida.
3. `Entorno de ejecución → Ejecutar todas`.

### Opción B — Local
```bash
pip install -r requirements.txt
python -m vatioplan.pipeline        # corre todo y deja resultados en salidas/
streamlit run app.py                # tablero en http://localhost:8501
```

### Informe con IA (opcional, gratis)
- Gemini: crear key en https://aistudio.google.com/apikey → `export GEMINI_API_KEY=...`
- Groq: crear key en https://console.groq.com/keys → `export GROQ_API_KEY=...`
- El modelo se cambia con `GEMINI_MODEL` / `GROQ_MODEL` si el proveedor publica uno más nuevo.
- Sin key, el informe se arma con una plantilla y todo sigue funcionando.

### Orquestación
- **Prefect:** `python flujo_prefect.py --una-vez` (prueba) · `python flujo_prefect.py` (queda programado a las 6:00).
  Interfaz: `prefect server start` → http://127.0.0.1:4200. Si el plan viola una regla, el flujo falla y **no** publica.
- **n8n:** importen `n8n_workflow.json`, cambien `/RUTA/A/vatioplan` y configuren las credenciales SMTP.
  En algunas instalaciones el nodo *Execute Command* viene desactivado por seguridad; hay que habilitarlo
  en la configuración de n8n autoalojado.

### Publicar el tablero (gratis)
Suban la carpeta a un repositorio de GitHub y en https://share.streamlit.io elijan `app.py`.
La API key va en *Settings → Secrets*.

## Notas técnicas
- **Retraso de publicación:** el precio más reciente tiene ~3 días; por eso el modelo pronostica hasta 7 días.
- **Versiones del precio:** XM re-liquida cada hora (TX1 → TX2 → TXR → TXF → TX3…); se usa la más reciente.
- **Datos simulados:** si no hay conexión, `cargar_datos` genera una serie sintética marcada como
  `SIMULADO` solo para probar el código. **Nunca presenten esas cifras como resultados.**
- **Supuesto de negocio:** el ahorro aplica cuando la empresa paga parte de su energía a precio de bolsa
  (parámetro `exposicion_bolsa`). Un usuario regulado con tarifa plana no ahorra moviendo cargas.
