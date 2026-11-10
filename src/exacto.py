"""
Solución exacta del SET COVER con programación lineal entera.

Modelo:
    min  sum_j c_j x_j
    s.a. sum_j a_ij x_j >= 1      para toda fila i = 1..m
         x_j in {0, 1}            para toda columna j = 1..n

Se implementa con PuLP y se resuelve con CBC (branch-and-cut), el solver
de código abierto que trae PuLP por defecto.
"""
import time

import numpy as np
import pulp

from instancia import InstanciaSCP


def resolver_exacto(inst: InstanciaSCP, limite_tiempo: float | None = None,
                    mostrar_log: bool = False) -> dict:
    t0 = time.perf_counter()

    # 1. Problema y variables de decisión
    modelo = pulp.LpProblem(f"SetCover_{inst.nombre}", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j + 1}", cat="Binary") for j in range(inst.n)]

    # 2. Función objetivo: costo total de las columnas seleccionadas
    modelo += pulp.lpDot(inst.costos.tolist(), x), "CostoTotal"

    # 3. Restricciones de cobertura: cada fila cubierta al menos una vez
    for i in range(inst.m):
        modelo += pulp.lpSum(x[j] for j in inst.cols_de_fila[i]) >= 1, f"Cubrir_fila_{i + 1}"

    t_construccion = time.perf_counter() - t0

    # 4. Solución con CBC
    solver = pulp.PULP_CBC_CMD(msg=mostrar_log, timeLimit=limite_tiempo)
    t1 = time.perf_counter()
    modelo.solve(solver)
    t_solucion = time.perf_counter() - t1

    x_val = np.array([round(v.value() or 0) for v in x], dtype=int)
    estado = pulp.LpStatus[modelo.status]
    # sol_status distingue entre óptimo probado (1) y solución factible al
    # alcanzar el límite de tiempo (2)
    estado_solucion = pulp.LpSolution[modelo.sol_status]
    optimo_probado = modelo.sol_status == pulp.LpSolutionOptimal

    return {
        "metodo": "Exacto (PuLP + CBC)",
        "instancia": inst.nombre,
        "estado": estado,
        "estado_solucion": estado_solucion,
        "optimo_probado": optimo_probado,
        "costo": inst.costo(x_val),
        "x": x_val,
        "columnas": (np.flatnonzero(x_val) + 1).tolist(),     # índices desde 1
        "num_columnas": int(x_val.sum()),
        "factible": inst.es_factible(x_val),
        "tiempo_construccion": t_construccion,
        "tiempo_solucion": t_solucion,
        "tiempo_total": t_construccion + t_solucion,
        "num_variables": len(modelo.variables()),
        "num_restricciones": len(modelo.constraints),
    }


if __name__ == "__main__":
    import sys
    from instancia import leer_orlib

    ruta = sys.argv[1] if len(sys.argv) > 1 else "../datos/scp41.txt"
    r = resolver_exacto(leer_orlib(ruta), mostrar_log=True)
    print(f"\n{r['instancia']}: estado={r['estado_solucion']}  costo={r['costo']:.0f}  "
          f"columnas={r['num_columnas']}  tiempo={r['tiempo_total']:.2f} s")
