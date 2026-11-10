"""
Algoritmo Genético (AG) para el SET COVER, basado en Beasley y Chu (1996).

Componentes:
    - Codificación : vector binario de n genes; gen j = 1 si se selecciona la columna j.
    - Aptitud      : costo total sum_j c_j x_j (se minimiza). Todos los individuos se
                     mantienen factibles gracias al operador de reparación.
    - Inicialización: para cada fila se elige al azar una columna que la cubra y luego
                     se eliminan las columnas redundantes.
    - Selección    : torneo binario (dos torneos, uno por cada padre).
    - Cruce        : operador de fusión (fusion crossover).
    - Mutación     : inversión de bits con tasa variable (crece con el número de hijos).
    - Reparación   : agrega columnas con menor costo por fila no cubierta y luego
                     elimina columnas redundantes empezando por las más costosas.
    - Reemplazo    : estado estacionario: el hijo reemplaza a un individuo con aptitud
                     peor que el promedio; no se aceptan hijos duplicados.
    - Parada       : número máximo de hijos, tiempo máximo o estancamiento.
"""
import math
import time
from dataclasses import dataclass, asdict

import numpy as np

from instancia import InstanciaSCP


@dataclass
class ParametrosAG:
    tam_poblacion: int = 100        # N individuos
    max_hijos: int = 50000          # criterio de parada 1
    max_sin_mejora: int = 15000     # criterio de parada 2 (estancamiento)
    tiempo_max: float = 120.0       # criterio de parada 3 (segundos)
    mf: float = 10.0                # mutación: número final de bits a mutar
    mc: float = 200.0               # mutación: hijos en que la tasa llega a mf/2
    mg: float = 2.0                 # mutación: pendiente del crecimiento
    semilla: int = 0
    registro_cada: int = 100        # cada cuántos hijos se guarda el historial


class AlgoritmoGenetico:
    def __init__(self, inst: InstanciaSCP, params: ParametrosAG | None = None):
        self.inst = inst
        self.p = params or ParametrosAG()
        self.rng = np.random.default_rng(self.p.semilla)
        self.A = inst.A
        self.c = inst.costos

    # ------------------------------------------------------------------ #
    # Operador de reparación (garantiza factibilidad y elimina redundancia)
    # ------------------------------------------------------------------ #
    def reparar(self, x: np.ndarray) -> np.ndarray:
        inst = self.inst
        cobertura = self.A[:, x.astype(bool)].sum(axis=1)       # veces que se cubre cada fila

        # Paso 1: cubrir filas descubiertas
        for i in np.flatnonzero(cobertura == 0):
            if cobertura[i] > 0:                                # ya la cubrió una columna agregada
                continue
            candidatas = inst.cols_de_fila[i]
            descubiertas = cobertura == 0
            nuevas = self.A[descubiertas][:, candidatas].sum(axis=0)   # filas nuevas que cubriría
            j = candidatas[np.argmin(self.c[candidatas] / nuevas)]
            x[j] = 1
            cobertura[inst.filas_de_col[j]] += 1

        # Paso 2: eliminar columnas redundantes (de mayor a menor costo)
        seleccionadas = np.flatnonzero(x)
        orden = seleccionadas[np.lexsort((self.rng.random(len(seleccionadas)),
                                          -self.c[seleccionadas]))]
        for j in orden:
            filas = inst.filas_de_col[j]
            if cobertura[filas].min() >= 2:                     # sin j todo sigue cubierto
                x[j] = 0
                cobertura[filas] -= 1
        return x

    # ------------------------------------------------------------------ #
    # Operadores genéticos
    # ------------------------------------------------------------------ #
    def individuo_inicial(self) -> np.ndarray:
        x = np.zeros(self.inst.n, dtype=np.int8)
        for i in range(self.inst.m):
            x[self.rng.choice(self.inst.cols_de_fila[i])] = 1
        return self.reparar(x)

    def torneo_binario(self, aptitud: np.ndarray) -> int:
        a, b = self.rng.choice(len(aptitud), size=2, replace=False)
        return a if aptitud[a] <= aptitud[b] else b

    def cruce_fusion(self, p1, p2, f1, f2) -> np.ndarray:
        """Si los padres coinciden se copia el gen; si no, se toma de p1 con
        probabilidad f2/(f1+f2), es decir, el padre más barato tiene más peso."""
        hijo = p1.copy()
        difieren = p1 != p2
        prob_p1 = f2 / (f1 + f2)
        de_p2 = difieren & (self.rng.random(len(p1)) >= prob_p1)
        hijo[de_p2] = p2[de_p2]
        return hijo

    def bits_a_mutar(self, t: int) -> int:
        p = self.p
        return math.ceil(p.mf / (1 + math.exp(-4 * p.mg * (t - p.mc) / p.mf)))

    def mutar(self, x: np.ndarray, t: int) -> np.ndarray:
        k = self.bits_a_mutar(t)
        posiciones = self.rng.choice(self.inst.n, size=k, replace=False)
        x[posiciones] = 1 - x[posiciones]
        return x

    # ------------------------------------------------------------------ #
    # Ciclo principal
    # ------------------------------------------------------------------ #
    def ejecutar(self, callback=None) -> dict:
        p = self.p
        t0 = time.perf_counter()

        # Población inicial
        poblacion = [self.individuo_inicial() for _ in range(p.tam_poblacion)]
        aptitud = np.array([self.c @ x for x in poblacion], dtype=float)
        firmas = {x.tobytes() for x in poblacion}

        mejor_idx = int(np.argmin(aptitud))
        mejor_x, mejor_f = poblacion[mejor_idx].copy(), aptitud[mejor_idx]
        hijo_mejora, tiempo_mejora = 0, time.perf_counter() - t0
        historial = [(0, mejor_f, aptitud.mean(), time.perf_counter() - t0)]
        duplicados = 0
        motivo = "max_hijos"

        t = 0
        while t < p.max_hijos:
            t += 1
            # Selección
            i1 = self.torneo_binario(aptitud)
            i2 = self.torneo_binario(aptitud)
            while i2 == i1:
                i2 = self.torneo_binario(aptitud)
            # Cruce + mutación + reparación
            hijo = self.cruce_fusion(poblacion[i1], poblacion[i2], aptitud[i1], aptitud[i2])
            hijo = self.mutar(hijo, t)
            hijo = self.reparar(hijo)
            f_hijo = float(self.c @ hijo)

            # Reemplazo (estado estacionario, sin duplicados)
            firma = hijo.tobytes()
            if firma in firmas:
                duplicados += 1
            else:
                peores = np.flatnonzero(aptitud > aptitud.mean())
                if len(peores) == 0:
                    peores = np.arange(len(aptitud))
                k = self.rng.choice(peores)
                firmas.discard(poblacion[k].tobytes())
                poblacion[k], aptitud[k] = hijo, f_hijo
                firmas.add(firma)
                if f_hijo < mejor_f:
                    mejor_x, mejor_f = hijo.copy(), f_hijo
                    hijo_mejora, tiempo_mejora = t, time.perf_counter() - t0

            if t % p.registro_cada == 0:
                historial.append((t, mejor_f, aptitud.mean(), time.perf_counter() - t0))
                if callback:
                    callback(t, mejor_f, aptitud.mean())

            if t - hijo_mejora >= p.max_sin_mejora:
                motivo = "estancamiento"
                break
            if time.perf_counter() - t0 >= p.tiempo_max:
                motivo = "tiempo_max"
                break

        tiempo = time.perf_counter() - t0
        if historial[-1][0] != t:
            historial.append((t, mejor_f, aptitud.mean(), tiempo))

        return {
            "metodo": "Algoritmo Genético",
            "instancia": self.inst.nombre,
            "costo": float(mejor_f),
            "x": mejor_x.astype(int),
            "columnas": (np.flatnonzero(mejor_x) + 1).tolist(),
            "num_columnas": int(mejor_x.sum()),
            "factible": self.inst.es_factible(mejor_x),
            "tiempo_total": tiempo,
            "hijos_generados": t,
            "hijo_mejor": hijo_mejora,
            "tiempo_mejor": tiempo_mejora,
            "duplicados": duplicados,
            "motivo_parada": motivo,
            "historial": np.array(historial),   # columnas: hijo, mejor, promedio, tiempo
            "parametros": asdict(p),
        }


def greedy_chvatal(inst: InstanciaSCP) -> dict:
    """Heurística voraz de Chvátal (1979), usada solo como referencia."""
    t0 = time.perf_counter()
    cubiertas = np.zeros(inst.m, dtype=bool)
    x = np.zeros(inst.n, dtype=int)
    while not cubiertas.all():
        nuevas = inst.A[~cubiertas].sum(axis=0)
        razon = np.where(nuevas > 0, inst.costos / np.maximum(nuevas, 1), np.inf)
        j = int(np.argmin(razon))
        x[j] = 1
        cubiertas |= inst.A[:, j]
    # eliminar redundancia
    cobertura = inst.A[:, x.astype(bool)].sum(axis=1)
    for j in sorted(np.flatnonzero(x), key=lambda j: -inst.costos[j]):
        filas = inst.filas_de_col[j]
        if cobertura[filas].min() >= 2:
            x[j] = 0
            cobertura[filas] -= 1
    return {"metodo": "Greedy (Chvátal)", "costo": inst.costo(x), "x": x,
            "tiempo_total": time.perf_counter() - t0, "factible": inst.es_factible(x)}


if __name__ == "__main__":
    import sys
    from instancia import leer_orlib

    ruta = sys.argv[1] if len(sys.argv) > 1 else "../datos/scp41.txt"
    inst = leer_orlib(ruta)
    r = AlgoritmoGenetico(inst, ParametrosAG(semilla=1)).ejecutar(
        callback=lambda t, b, m: print(f"hijo {t:6d}  mejor={b:.0f}  promedio={m:.1f}")
        if t % 1000 == 0 else None)
    print(f"\n{r['instancia']}: costo={r['costo']:.0f}  tiempo={r['tiempo_total']:.1f} s  "
          f"parada={r['motivo_parada']}  factible={r['factible']}")
