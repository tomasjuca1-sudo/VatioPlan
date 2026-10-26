"""
flotas.py
Carga inteligente de flotas electricas (buses, furgonetas de reparto, montacargas).

Idea: los vehiculos llegan al patio, se conectan y deben salir con la bateria cargada.
Mientras estan conectados, la carga se puede mover a las horas mas baratas.

Ventana de planeacion: 24 horas que empiezan a la `hora_inicio` de un dia (por defecto
12:00) y terminan a la misma hora del dia siguiente. Asi una carga nocturna (llega a las
21:00 y sale a las 05:00) queda completa dentro de una sola ventana.

Dos formas de cargar:
  * Carga inmediata (operacion actual): cada vehiculo carga a toda potencia apenas llega,
    hasta llenarse, respetando la potencia maxima del patio.
  * VatioPlan: programacion lineal (PuLP) que minimiza el costo de la energia y cumple:
      R1. Cada grupo recibe toda la energia que necesita antes de su hora de salida.
      R2. Un grupo solo carga mientras esta conectado (entre llegada y salida).
      R3. Cada grupo no supera la potencia de sus cargadores (cantidad x kW por cargador).
      R4. La suma de todos los grupos no supera la potencia maxima del patio.

Los vehiculos de un mismo grupo son iguales, por eso se planea por grupo y la energia
se reparte por igual entre sus vehiculos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None

HORA_INICIO_VENTANA = 12
EFICIENCIA_CARGA = 0.90   # energia que llega a la bateria / energia tomada de la red


@dataclass
class GrupoVehiculos:
    nombre: str
    cantidad: int
    bateria_kwh: float
    cargador_kw: float
    hora_llegada: int
    hora_salida: int
    carga_llegada_pct: float
    carga_salida_pct: float

    def energia_red_kwh(self) -> float:
        """Energia que hay que tomar de la red para todo el grupo."""
        energia_bateria = self.cantidad * self.bateria_kwh * (self.carga_salida_pct - self.carga_llegada_pct) / 100
        return max(energia_bateria, 0.0) / EFICIENCIA_CARGA

    def potencia_maxima_kw(self) -> float:
        return self.cantidad * self.cargador_kw

    def indice_llegada(self, hora_inicio: int) -> int:
        return (self.hora_llegada - hora_inicio) % 24

    def indice_salida(self, hora_inicio: int) -> int:
        indice = (self.hora_salida - hora_inicio) % 24
        return 24 if indice == 0 else indice

    def horas_conectado(self, hora_inicio: int) -> list[int]:
        """Posiciones de la ventana (0 a 23) en las que el grupo puede cargar."""
        return list(range(self.indice_llegada(hora_inicio), self.indice_salida(hora_inicio)))


def grupos_desde_tabla(tabla: pd.DataFrame) -> list[GrupoVehiculos]:
    """Convierte la tabla editable de la app en una lista de grupos. Ignora filas incompletas."""
    grupos = []
    for _, fila in tabla.dropna().iterrows():
        if int(fila["Cantidad"]) <= 0:
            continue
        grupos.append(GrupoVehiculos(
            nombre=str(fila["Grupo"]),
            cantidad=int(fila["Cantidad"]),
            bateria_kwh=float(fila["Batería (kWh)"]),
            cargador_kw=float(fila["Cargador (kW)"]),
            hora_llegada=int(fila["Llega (hora)"]),
            hora_salida=int(fila["Sale (hora)"]),
            carga_llegada_pct=float(fila["Carga al llegar (%)"]),
            carga_salida_pct=float(fila["Carga al salir (%)"]),
        ))
    return grupos


def validar_grupos(grupos: list[GrupoVehiculos], hora_inicio: int) -> list[str]:
    """Errores que impiden planear. Lista vacia si todo esta bien."""
    errores = []
    for g in grupos:
        horas = g.horas_conectado(hora_inicio)
        if g.indice_llegada(hora_inicio) >= g.indice_salida(hora_inicio):
            errores.append(f"{g.nombre}: la salida debe quedar despues de la llegada dentro de la ventana "
                           f"que empieza a las {hora_inicio:02d}:00.")
            continue
        if g.carga_salida_pct > 100 or g.carga_llegada_pct < 0:
            errores.append(f"{g.nombre}: los porcentajes de carga deben estar entre 0 y 100.")
        energia_posible = g.potencia_maxima_kw() * len(horas)
        texto_horas = "1 hora" if len(horas) == 1 else f"{len(horas)} horas"
        if g.energia_red_kwh() > energia_posible + 1e-6:
            errores.append(f"{g.nombre}: necesita {g.energia_red_kwh():,.0f} kWh, pero con sus cargadores solo "
                           f"alcanza a tomar {energia_posible:,.0f} kWh en {texto_horas} conectado. "
                           "Aumente la potencia del cargador, el tiempo conectado o baje la carga al salir.")
    return errores


def precios_ventana(serie: pd.Series, fecha, hora_inicio: int) -> np.ndarray | None:
    """24 precios desde `fecha` a la `hora_inicio`. None si falta alguna hora."""
    inicio = pd.Timestamp(fecha).normalize() + pd.Timedelta(hours=hora_inicio)
    horas = pd.date_range(inicio, periods=24, freq="h")
    valores = serie.reindex(horas)
    if valores.isna().any():
        return None
    return valores.values


def ventanas_disponibles(serie: pd.Series, hora_inicio: int) -> list:
    """Fechas cuya ventana de 24 horas esta completa en la serie."""
    fechas = []
    for fecha in sorted(set(serie.index.date)):
        if precios_ventana(serie, fecha, hora_inicio) is not None:
            fechas.append(fecha)
    return fechas


def carga_inmediata(grupos: list[GrupoVehiculos], potencia_patio_kw: float, hora_inicio: int) -> pd.DataFrame:
    """Operacion actual: cargar a toda potencia apenas llegan. Si el patio se queda corto,
    tienen prioridad los grupos que salen primero."""
    plan = pd.DataFrame(0.0, index=range(24), columns=[g.nombre for g in grupos])
    pendiente = {g.nombre: g.energia_red_kwh() for g in grupos}
    orden = sorted(grupos, key=lambda g: g.indice_salida(hora_inicio))
    for h in range(24):
        disponible_patio = potencia_patio_kw
        for g in orden:
            if h not in g.horas_conectado(hora_inicio):
                continue
            potencia = min(g.potencia_maxima_kw(), pendiente[g.nombre], disponible_patio)
            plan.loc[h, g.nombre] = potencia
            pendiente[g.nombre] -= potencia
            disponible_patio -= potencia
    return plan


def optimizar_carga(precios: np.ndarray, grupos: list[GrupoVehiculos], potencia_patio_kw: float,
                    hora_inicio: int) -> pd.DataFrame | None:
    """Plan de minimo costo (kW por grupo y hora de la ventana). None si no hay plan posible."""
    modelo = pulp.LpProblem("carga_flota", pulp.LpMinimize)
    potencia = {}
    for i, g in enumerate(grupos):
        for h in g.horas_conectado(hora_inicio):                                         # R2
            potencia[g.nombre, h] = pulp.LpVariable(f"p_{i}_{h}", lowBound=0,
                                                    upBound=g.potencia_maxima_kw())      # R3
    modelo += pulp.lpSum(precios[h] * potencia[nombre, h] for (nombre, h) in potencia)
    for g in grupos:
        modelo += pulp.lpSum(potencia[g.nombre, h] for h in g.horas_conectado(hora_inicio)) \
            == g.energia_red_kwh()                                                       # R1
    for h in range(24):
        en_hora = [potencia[nombre, hh] for (nombre, hh) in potencia if hh == h]
        if en_hora:
            modelo += pulp.lpSum(en_hora) <= potencia_patio_kw                           # R4
    modelo.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[modelo.status] != "Optimal":
        return None
    plan = pd.DataFrame(0.0, index=range(24), columns=[g.nombre for g in grupos])
    for (nombre, h), variable in potencia.items():
        plan.loc[h, nombre] = max(variable.value() or 0.0, 0.0)
    return plan


def costo_carga(plan: pd.DataFrame, precios: np.ndarray, exposicion_bolsa: float = 1.0) -> float:
    """Costo en COP: kWh tomados de la red en cada hora x precio x exposicion a bolsa."""
    return float(np.sum(plan.sum(axis=1).values * precios) * exposicion_bolsa)


def verificar_plan(plan: pd.DataFrame, grupos: list[GrupoVehiculos], potencia_patio_kw: float,
                   hora_inicio: int) -> pd.DataFrame:
    """Comprueba que el plan cumple las reglas R1 a R4 para cada grupo."""
    filas = []
    pico_patio = plan.sum(axis=1).max()
    for g in grupos:
        energia_entregada = plan[g.nombre].sum()
        fuera_de_horario = plan[g.nombre].drop(index=g.horas_conectado(hora_inicio)).sum()
        filas.append({
            "Grupo": g.nombre,
            "Energía requerida (kWh)": round(g.energia_red_kwh()),
            "Energía programada (kWh)": round(energia_entregada),
            "Sale con la carga pedida": "Sí" if energia_entregada >= g.energia_red_kwh() - 0.5 else "No",
            "Carga solo conectado": "Sí" if fuera_de_horario < 0.5 else "No",
            "Dentro de cargadores": "Sí" if plan[g.nombre].max() <= g.potencia_maxima_kw() + 0.5 else "No",
            "Dentro del límite del patio": "Sí" if pico_patio <= potencia_patio_kw + 0.5 else "No",
        })
    return pd.DataFrame(filas)


def rangos_ventana(plan_grupo: pd.Series, hora_inicio: int) -> str:
    """Horas de carga de un grupo como rangos de reloj, por ejemplo '01:00 a 05:00'."""
    horas = [h for h in range(24) if plan_grupo[h] > 0.5]
    if not horas:
        return "no carga"
    bloques = []
    inicio_bloque = horas[0]
    for i, h in enumerate(horas):
        es_ultimo = i == len(horas) - 1
        if es_ultimo or horas[i + 1] != h + 1:
            desde = (inicio_bloque + hora_inicio) % 24
            hasta = (h + 1 + hora_inicio) % 24
            bloques.append(f"{desde:02d}:00 a {hasta:02d}:00")
            if not es_ultimo:
                inicio_bloque = horas[i + 1]
    return ", ".join(bloques)


def backtest_flota(real: pd.Series, pred: pd.Series, grupos: list[GrupoVehiculos],
                   potencia_patio_kw: float, hora_inicio: int) -> pd.DataFrame:
    """Para cada noche del periodo de prueba: plan con el pronostico, pagado con el precio real.
    Se compara contra la carga inmediata y contra el plan perfecto (precio conocido)."""
    plan_inmediato = carga_inmediata(grupos, potencia_patio_kw, hora_inicio)
    filas = []
    for fecha in ventanas_disponibles(real, hora_inicio):
        precios_reales = precios_ventana(real, fecha, hora_inicio)
        precios_pronostico = precios_ventana(pred, fecha, hora_inicio)
        if precios_pronostico is None:
            continue
        plan_pronostico = optimizar_carga(precios_pronostico, grupos, potencia_patio_kw, hora_inicio)
        plan_perfecto = optimizar_carga(precios_reales, grupos, potencia_patio_kw, hora_inicio)
        if plan_pronostico is None or plan_perfecto is None:
            continue
        filas.append({
            "fecha": pd.Timestamp(fecha),
            "costo_inmediata": costo_carga(plan_inmediato, precios_reales),
            "costo_con_pronostico": costo_carga(plan_pronostico, precios_reales),
            "costo_perfecto": costo_carga(plan_perfecto, precios_reales),
        })
    if not filas:
        return pd.DataFrame(columns=["costo_inmediata", "costo_con_pronostico", "costo_perfecto"])
    return pd.DataFrame(filas).set_index("fecha")


def grupos_ejemplo() -> pd.DataFrame:
    """Flota de ejemplo para la tabla editable de la app."""
    return pd.DataFrame([
        {"Grupo": "Buses eléctricos", "Cantidad": 10, "Batería (kWh)": 300, "Cargador (kW)": 80,
         "Llega (hora)": 21, "Sale (hora)": 5, "Carga al llegar (%)": 30, "Carga al salir (%)": 95},
        {"Grupo": "Furgonetas de reparto", "Cantidad": 15, "Batería (kWh)": 75, "Cargador (kW)": 22,
         "Llega (hora)": 19, "Sale (hora)": 7, "Carga al llegar (%)": 25, "Carga al salir (%)": 90},
        {"Grupo": "Montacargas", "Cantidad": 8, "Batería (kWh)": 40, "Cargador (kW)": 10,
         "Llega (hora)": 18, "Sale (hora)": 6, "Carga al llegar (%)": 20, "Carga al salir (%)": 100},
    ])
