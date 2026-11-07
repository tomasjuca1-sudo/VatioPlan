"""
informe.py: IA generativa: convierte los números del pipeline en un informe claro
para el jefe de planta (fábrica) o el jefe de patio (flota eléctrica).

Proveedores gratuitos soportados (se usa el primero que tenga API key):
  * Google Gemini  → variable de entorno GEMINI_API_KEY (https://aistudio.google.com/apikey)
  * Groq (gpt-oss) → variable de entorno GROQ_API_KEY   (https://console.groq.com/keys)
Si no hay ninguna key, se genera un informe con plantilla (el pipeline nunca se detiene).

Buenas prácticas aplicadas:
  * El LLM recibe SOLO los números calculados por el pipeline (JSON) y se le prohíbe inventar cifras.
  * Las decisiones las toman el modelo y las reglas; el LLM solo redacta y prioriza.
"""
from __future__ import annotations

import json
import os
import re

import requests

GEMINI_MODELO = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GROQ_MODELO = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

LECTOR = {"Fábrica manufacturera": ("el jefe de planta", "equipo", "horario recomendado de operación"),
          "Flota eléctrica": ("el jefe de patio de la flota", "grupo de vehículos", "horario recomendado de carga")}

INSTRUCCIONES = """Eres el asistente de energía de un cliente industrial en Colombia ({tipo}).
Con los DATOS (JSON) escribe un informe breve en español para {lector}:
1. Titular de una línea con el ahorro esperado (en COP y %) frente a "{referencia}".
2. Qué hacer: una línea por {elemento} con el {accion}.
3. Horas a evitar (las más caras) y horas a aprovechar (las más baratas).
4. Confianza: ahorro real medido en backtest frente al máximo posible.
5. Riesgos y supuestos (exposición a bolsa, límite de potencia, reglas verificadas, fuente de datos, supuestos del caso).
Reglas: usa ÚNICAMENTE cifras que aparezcan en los DATOS. No inventes números.
Máximo 230 palabras, formato Markdown con viñetas cortas.
Estilo: no uses punto y coma, punto medio (·), raya (—) ni guion largo (–). Escribe los rangos con "a"
(por ejemplo, "01:00 a 05:00")."""


def _gemini(prompt: str, key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODELO}:generateContent"
    r = requests.post(url, headers={"x-goog-api-key": key},
                      json={"contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.2}}, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _groq(prompt: str, key: str) -> str:
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": GROQ_MODELO, "temperature": 0.2,
                            "messages": [{"role": "user", "content": prompt}]}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def plantilla(d: dict) -> str:
    """Informe sin LLM: mismas cifras, redacción fija."""
    _, elemento, accion = LECTOR.get(d.get("tipo_usuario"), LECTOR["Fábrica manufacturera"])
    lin = [f"### ⚡ Ahorro esperado: ${d['ahorro_cop']:,.0f} COP ({d['ahorro_pct']:.1f}%) en "
           f"{d['ventanas_planeadas']} días frente a {d['plan_de_referencia'].lower()}",
           "", f"**{d['usuario']}**. {accion.capitalize()} (primera ventana):"]
    lin += [f"- **{c}**: {h}" for c, h in d["horario_recomendado_primera_ventana"].items()]
    lin += ["", f"**Evitar:** {', '.join(d['horas_mas_caras'])} (hasta {d['precio_max']:,.0f} COP/kWh). "
                f"**Aprovechar:** {', '.join(d['horas_mas_baratas'])} (desde {d['precio_min']:,.0f} COP/kWh).",
            "", "**Confianza y supuestos:**",
            f"- Ahorro real en backtest: {d['ahorro_real_backtest_pct']:.1f}% "
            f"(máximo posible {d['ahorro_maximo_backtest_pct']:.1f}%, {d['dias_backtest']} días con precio real).",
            f"- El pronóstico identifica {d['acierto_horas_baratas']:.0%} de las horas baratas "
            f"(error medio {d['mae_modelo']:,.0f} COP/kWh).",
            f"- Exposición a precio de bolsa: {d['exposicion_bolsa']:.0%}.",
            f"- Límite de potencia: {d['limite_potencia_kw']:,.0f} kW.",
            f"- Reglas de ingeniería: {d['reglas']}.",
            f"- Fuente de datos: {d['fuente_datos']}."]
    return "\n".join(lin)


def limpiar_estilo(texto: str) -> str:
    """Quita del texto del LLM los signos que el informe no usa: rangos con guion largo pasan a "a",
    y el punto medio y la raya pasan a coma."""
    texto = re.sub(r"([A-Za-z]?\d)\s?[–—]\s?([A-Za-z]?\d)", r"\1 a \2", texto)
    texto = re.sub(r"\s*[—–]\s*", ", ", texto)
    texto = re.sub(r"\s*·\s*", ", ", texto)
    return texto


def generar_informe(datos: dict) -> tuple[str, str]:
    """Devuelve (texto_markdown, proveedor_usado)."""
    lector, elemento, accion = LECTOR.get(datos.get("tipo_usuario"), LECTOR["Fábrica manufacturera"])
    instrucciones = INSTRUCCIONES.format(tipo=datos.get("tipo_usuario", "industria"), lector=lector,
                                         referencia=datos.get("plan_de_referencia", "la operación actual"),
                                         elemento=elemento, accion=accion)
    prompt = f"{instrucciones}\n\nDATOS:\n{json.dumps(datos, ensure_ascii=False, indent=1, default=str)}"
    for proveedor, var, fn in [("Gemini", "GEMINI_API_KEY", _gemini), ("Groq", "GROQ_API_KEY", _groq)]:
        key = os.getenv(var)
        if key:
            try:
                return limpiar_estilo(fn(prompt, key)), proveedor
            except Exception as e:
                print(f"⚠️  {proveedor} falló ({e}); probando la siguiente opción.")
    return plantilla(datos), "plantilla (sin API key)"
