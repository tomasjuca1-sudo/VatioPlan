# ⚡ VatioPlan

**Mover el consumo eléctrico flexible a las horas más baratas de la bolsa de energía en Colombia.**
Proyecto final: Herramientas de Inteligencia Artificial, Universidad de los Andes (2026).

VatioPlan pronostica el **precio de bolsa horario** con machine learning, reprograma el consumo flexible
respetando **reglas de ingeniería** y entrega un **informe redactado con IA generativa**. Todo corre solo cada mañana.

## Dos tipos de usuario, dos casos reales de referencia

| Usuario | Caso de referencia (`config/`) | Qué se mueve | Reglas |
|---|---|---|---|
| 🏭 **Fábrica manufacturera** | Cementos Argos, planta Rioclaro (Antioquia): 4.000 t/d de clínker, 4.245 t/d de cemento | Molino de crudo, 2 molinos de cemento, molino de carbón, trituradora (los silos guardan el material) | R1 horas diarias, R2 ventana permitida, R3 bloque continuo, R4 potencia contratada, R5 corrida mínima |
| 🚌 **Flota eléctrica** | Patio Fontibón I (Refugio), SITP Bogotá (Gran Américas Fontibón I): 120 buses BYD, 56 puntos dobles de 150 kW | La carga nocturna de cada grupo de buses | R1 sale con la carga pedida, R2 solo conectado, R3 potencia del cargador, R4 límite del patio |

Los dos usan **el mismo pronóstico**. Cada archivo de `config/` trae sus **fuentes** y sus **supuestos**
(las potencias por equipo y las baterías no son públicas y se estiman con fuentes técnicas citadas).
Para otro cliente basta copiar el archivo, cambiar los valores y subirlo en el tablero.

```
SIMEM/XM + NOAA ─▶ Datos ─▶ Modelo ML ─▶ Pronóstico ─┬▶ Fábrica: optimización + R1 a R5 ─┬▶ IA generativa ─▶ Informes / Tablero
                                                       └▶ Flota:   optimización + R1 a R4 ─┘
                     ▲ orquestado todos los días ~6:00 a. m. (GitHub Actions, Prefect o n8n) ▲
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `vatioplan/datos.py` | Descarga automática: precio horario (SIMEM `EC6945`) por bloques con caché, pausas y reintentos ante bloqueos 403; embalses y aportes (API XM, incremental); ONI (NOAA). |
| `vatioplan/features.py` | Variables del modelo, todas rezagadas ≥ 7 días (sin fuga de datos). |
| `vatioplan/modelo.py` | Gradient boosting (scikit-learn) que corrige el promedio de la misma hora en las últimas 4 semanas. Se compara contra el ingenuo, el promedio de 4 semanas y las versiones anteriores. |
| `vatioplan/planta.py` | Fábrica: programación entera (PuLP), reglas R1 a R5 y validación. |
| `vatioplan/flotas.py` | Flota: programación lineal (PuLP) de la carga por grupo, reglas R1 a R4, carga inmediata como referencia. |
| `vatioplan/perfiles.py` | Interfaz común de los dos usuarios: planeación, backtest, comparación de modelos y resumen para la IA. |
| `vatioplan/informe.py` | Informe con Gemini o Groq (gratis), adaptado a cada usuario; plantilla si no hay API key. |
| `vatioplan/pipeline.py` | Une todos los pasos y escribe `salidas/fabrica/`, `salidas/flota/` y `salidas/estado.json`. |
| `config/fabrica_argos_rioclaro.json` | Caso fábrica, con fuentes y supuestos. |
| `config/flota_fontibon_refugio.json` | Caso flota, con fuentes y supuestos. |
| `config/planta_ejemplo.json` | Planta pequeña de plásticos (ejemplo didáctico original). |
| `app.py` | Tablero Streamlit con selector de tipo de usuario. |
| `notebooks/VatioPlan_completo.ipynb` | Recorrido completo, paso a paso, con gráficas. |
| `.github/workflows/vatioplan_diario.yml` | Orquestación en la nube (GitHub Actions, 6:15 a. m. Bogotá). |
| `flujo_prefect.py` | Orquestación con Prefect (tareas con reintentos; no publica si se viola una regla). |
| `n8n_workflow.json` | Orquestación visual: lanza GitHub Actions y envía un informe a cada usuario por correo. |

## Cómo correrlo

```bash
pip install -r requirements.txt
python -m vatioplan.pipeline                 # los dos usuarios; resultados en salidas/
python -m vatioplan.pipeline --perfil flota  # solo uno
streamlit run app.py                         # tablero en http://localhost:8501
```
La primera corrida descarga 2023 → hoy desde SIMEM (unos minutos). Si SIMEM bloquea a mitad de camino,
vuelvan a correrlo: los bloques ya descargados quedan en `data/simem_bloques/`.

**Colab:** abran el notebook en https://colab.research.google.com; la primera celda clona el repositorio.

## Pasos que faltan para dejar la orquestación andando

1. **Subir esta versión al repositorio.** Reemplacen todo el contenido (incluidas las carpetas ocultas `.github/` y `.streamlit/`).
   Con la página de GitHub: *Add file → Upload files* y arrastren las carpetas completas.
2. **Primera descarga desde un computador en Colombia** (SIMEM bloquea más a los servidores en la nube):
   `python -m vatioplan.pipeline`, y luego suban la carpeta `data/` al repositorio. Así GitHub Actions y Streamlit
   solo descargan los últimos días.
3. **Key de IA (opcional):** creen una key gratis en https://aistudio.google.com/apikey.
   - GitHub: *Settings → Secrets and variables → Actions → New repository secret* → `GEMINI_API_KEY`.
   - Streamlit Cloud: *Settings → Secrets* con el contenido de `.streamlit/secrets.toml.example`.
4. **Activar GitHub Actions:** pestaña *Actions* → *VatioPlan diario* → *Run workflow* para probar.
   Revisen que el paso “Verificar que los datos son reales” quede en verde. Desde ahí corre solo a las 6:15 a. m.
   Si *Settings → Actions → General → Workflow permissions* está en “Read”, cámbienlo a “Read and write”.
5. **Streamlit:** *Reboot app* en https://share.streamlit.io para limpiar el caché con el código nuevo.
6. **Prefect (opcional, local):** `pip install -r requirements-orquestacion.txt` → `python flujo_prefect.py --una-vez`
   para probar y `python flujo_prefect.py` para dejarlo programado. Interfaz: `prefect server start`.
7. **n8n (opcional):** importen `n8n_workflow.json`; cambien el token de GitHub (permiso *Actions: write*), el nombre del
   repositorio si cambia, los correos de destino y las credenciales SMTP. Si usan n8n, desactiven el `schedule`
   del workflow de GitHub para no correr dos veces.

## Informe con IA (opcional, gratis)
- Gemini: `GEMINI_API_KEY`, Groq: `GROQ_API_KEY` (respaldo).
- El modelo se cambia con `GEMINI_MODEL` / `GROQ_MODEL` si el proveedor publica uno más nuevo.
- Sin key, el informe se arma con una plantilla y todo sigue funcionando.
- En el tablero, el informe usa los **parámetros de la barra lateral** (exposición, potencia, vehículos).

## Notas técnicas
- **Retraso de publicación:** el precio más reciente tiene ~3 días; por eso el modelo pronostica hasta 7 días.
- **Versiones del precio:** XM re-liquida cada hora (TX1 → TX2 → TXR → TXF → TX3…); se usa la más reciente.
- **API XM:** la respuesta trae el valor en `{"Id": "Sistema", "Value": ...}` (corregido; antes llegaba vacío).
- **Datos simulados:** si no hay conexión, `cargar_datos` genera una serie sintética marcada como
  `SIMULADO` solo para probar el código. **Nunca presenten esas cifras como resultados.**
- **Supuesto de negocio:** el ahorro aplica a la fracción de energía pagada a precio de bolsa
  (`exposicion_bolsa`). Un usuario regulado con tarifa plana no ahorra moviendo consumo.
- **Ahorro estimado vs. real:** el costo del día usa el pronóstico; el ahorro real en backtest planea con el
  pronóstico y paga con el precio real. La cifra de desempeño es la segunda.
