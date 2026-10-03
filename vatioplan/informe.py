"""
informe.py — IA generativa: convierte los números del pipeline en un informe claro
para el jefe de planta.

Proveedores gratuitos soportados (se usa el primero que tenga API key):
  * Google Gemini  → variable de entorno GEMINI_API_KEY (https://aistudio.google.com/apikey)
  * Groq (Llama)   → variable de entorno GROQ_API_KEY   (https://console.groq.com/keys)
Si no hay ninguna key, se genera un informe con plantilla (el pipeline nunca se detiene).

Buenas prácticas aplicadas:
  * El LLM recibe SOLO los números calculados por el pipeline (JSON) y se le prohíbe inventar cifras.
  * Las decisiones las toman el modelo y las reglas; el LLM solo redacta y prioriza.
"""
from __future__ import annotations

import json
import os

import requests

GEMINI_MODELO = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")   # cámbienlo si Google publica uno nuevo
GROQ_MODELO = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

INSTRUCCIONES = """Eres el asistente de energía de una planta industrial en Colombia.
Con los DATOS (JSON) escribe un informe breve en español para el jefe de planta:
1. Titular de una línea con el ahorro esperado (en COP y %).
2. Qué hacer: una línea por carga con el horario recomendado.
3. Horas a evitar (las más caras) y por qué importan.
4. Riesgos y supuestos (confianza del pronóstico, exposición a bolsa, reglas verificadas).
Reglas: usa ÚNICAMENTE cifras que aparezcan en los DATOS; no inventes números;
máximo 220 palabras; formato Markdown con viñetas cortas."""


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


def _plantilla(d: dict) -> str:
    lin = [f"### ⚡ Ahorro esperado: ${d['ahorro_cop']:,.0f} COP ({d['ahorro_pct']:.1f}%) en {d['dias']} días",
           "", "**Qué hacer:**"]
    lin += [f"- **{c}**: {h}" for c, h in d["horario_recomendado"].items()]
    lin += ["", f"**Evitar:** {', '.join(d['horas_mas_caras'])} (precio pronosticado hasta "
                f"{d['precio_max']:,.0f} COP/kWh vs. mínimo {d['precio_min']:,.0f}).",
            "", "**Supuestos y riesgos:**",
            f"- Pronóstico con error medio (MAE) de {d['mae_modelo']:,.0f} COP/kWh; "
            f"identifica {d['acierto_horas_baratas']:.0%} de las horas baratas.",
            f"- Exposición a precio de bolsa: {d['exposicion_bolsa']:.0%}.",
            f"- Reglas de ingeniería: {d['reglas']}.",
            f"- Fuente de datos: {d['fuente']}."]
    return "\n".join(lin)


def generar_informe(datos: dict) -> tuple[str, str]:
    """Devuelve (texto_markdown, proveedor_usado)."""
    prompt = f"{INSTRUCCIONES}\n\nDATOS:\n{json.dumps(datos, ensure_ascii=False, indent=1, default=str)}"
    for proveedor, var, fn in [("Gemini", "GEMINI_API_KEY", _gemini), ("Groq", "GROQ_API_KEY", _groq)]:
        key = os.getenv(var)
        if key:
            try:
                return fn(prompt, key), proveedor
            except Exception as e:
                print(f"⚠️  {proveedor} falló ({e}); probando la siguiente opción.")
    return _plantilla(datos), "plantilla (sin API key)"
