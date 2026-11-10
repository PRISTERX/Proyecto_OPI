"""
Experimentos computacionales: comparación entre el método exacto y el AG.

    E1. Instancias OR-Library (conjuntos 4 y 5): exacto vs. AG (10 semillas) vs. greedy.
    E2. Escalabilidad: instancias aleatorias de tamaño creciente.
    E3. Sensibilidad de parámetros del AG (tamaño de población y mf).

Uso:
    python experimentos.py e1      (o e2, e3, todos)
Los resultados se guardan en ../resultados/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from instancia import leer_orlib, generar_aleatoria, instancias_disponibles, OPTIMOS_LITERATURA
from exacto import resolver_exacto
from ga import AlgoritmoGenetico, ParametrosAG, greedy_chvatal

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "datos"
RES = BASE / "resultados"
RES.mkdir(exist_ok=True)

SEMILLAS = list(range(1, 11))       # 10 ejecuciones independientes del AG


def gap(valor, optimo):
    return 100.0 * (valor - optimo) / optimo


def experimento_1():
    filas_exacto, filas_ag, historiales = [], [], {}
    for ruta in instancias_disponibles(DATOS):
        inst = leer_orlib(ruta)
        ex = resolver_exacto(inst)
        gr = greedy_chvatal(inst)
        filas_exacto.append({
            "instancia": inst.nombre, "m": inst.m, "n": inst.n,
            "densidad": round(inst.densidad, 2),
            "optimo_literatura": OPTIMOS_LITERATURA.get(inst.nombre),
            "costo_exacto": ex["costo"], "estado": ex["estado_solucion"],
            "columnas_exacto": ex["num_columnas"],
            "tiempo_exacto": ex["tiempo_total"],
            "costo_greedy": gr["costo"], "tiempo_greedy": gr["tiempo_total"],
            "seleccion_exacto": " ".join(map(str, ex["columnas"])),
        })
        for s in SEMILLAS:
            r = AlgoritmoGenetico(inst, ParametrosAG(semilla=s)).ejecutar()
            filas_ag.append({
                "instancia": inst.nombre, "semilla": s, "costo_ag": r["costo"],
                "columnas_ag": r["num_columnas"], "tiempo_ag": r["tiempo_total"],
                "hijos": r["hijos_generados"], "hijo_mejor": r["hijo_mejor"],
                "tiempo_mejor": r["tiempo_mejor"], "duplicados": r["duplicados"],
                "parada": r["motivo_parada"], "factible": r["factible"],
                "gap": gap(r["costo"], ex["costo"]),
            })
            historiales[f"{inst.nombre}_s{s}"] = r["historial"].tolist()
            print(f"E1 {inst.nombre} s={s}: AG={r['costo']:.0f} exacto={ex['costo']:.0f} "
                  f"t={r['tiempo_total']:.1f}s", flush=True)

    ex_df = pd.DataFrame(filas_exacto)
    ag_df = pd.DataFrame(filas_ag)
    ex_df.to_csv(RES / "e1_exacto.csv", index=False)
    ag_df.to_csv(RES / "e1_ag_corridas.csv", index=False)
    (RES / "e1_historiales.json").write_text(json.dumps(historiales))

    resumen = ag_df.groupby("instancia", sort=False).agg(
        ag_mejor=("costo_ag", "min"), ag_promedio=("costo_ag", "mean"),
        ag_peor=("costo_ag", "max"), ag_desv=("costo_ag", "std"),
        gap_promedio=("gap", "mean"), gap_mejor=("gap", "min"),
        tiempo_ag_prom=("tiempo_ag", "mean"),
        tiempo_mejor_prom=("tiempo_mejor", "mean"),
        exitos=("gap", lambda g: int((g <= 1e-9).sum())),
    ).reset_index()
    tabla = ex_df.drop(columns="seleccion_exacto").merge(resumen, on="instancia")
    tabla.to_csv(RES / "e1_resumen.csv", index=False)
    print(tabla.round(2).to_string())


def experimento_2():
    tamanos = [(200, 2000), (400, 4000), (600, 6000), (800, 8000), (1000, 10000)]
    filas = []
    for m, n in tamanos:
        inst = generar_aleatoria(m, n, densidad=0.02, costo_max=100, semilla=2026,
                                 nombre=f"rand{m}x{n}")
        ex = resolver_exacto(inst, limite_tiempo=180)
        costos, tiempos = [], []
        for s in [1, 2, 3]:
            r = AlgoritmoGenetico(inst, ParametrosAG(semilla=s, tiempo_max=60)).ejecutar()
            costos.append(r["costo"]); tiempos.append(r["tiempo_total"])
        filas.append({
            "instancia": inst.nombre, "m": m, "n": n,
            "costo_exacto": ex["costo"], "estado": ex["estado_solucion"],
            "optimo_probado": ex["optimo_probado"], "tiempo_exacto": ex["tiempo_total"],
            "ag_mejor": min(costos), "ag_promedio": float(np.mean(costos)),
            "tiempo_ag_prom": float(np.mean(tiempos)),
            "gap_promedio": gap(float(np.mean(costos)), ex["costo"]),
        })
        print(filas[-1], flush=True)
    pd.DataFrame(filas).to_csv(RES / "e2_escalabilidad.csv", index=False)


def experimento_3():
    filas = []
    configuraciones = [(pob, mf) for pob in (50, 100, 200) for mf in (5, 10, 20)]
    for nombre in ("scp49", "scp57"):
        inst = leer_orlib(DATOS / f"{nombre}.txt")
        opt = OPTIMOS_LITERATURA[nombre]
        for pob, mf in configuraciones:
            for s in [1, 2, 3, 4, 5]:
                r = AlgoritmoGenetico(inst, ParametrosAG(tam_poblacion=pob, mf=mf,
                                                         semilla=s)).ejecutar()
                filas.append({"instancia": nombre, "poblacion": pob, "mf": mf,
                              "semilla": s, "costo": r["costo"],
                              "gap": gap(r["costo"], opt), "tiempo": r["tiempo_total"]})
                print(filas[-1], flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(RES / "e3_sensibilidad.csv", index=False)
    print(df.groupby(["instancia", "poblacion", "mf"])[["gap", "tiempo"]].mean().round(2))


if __name__ == "__main__":
    cual = sys.argv[1] if len(sys.argv) > 1 else "todos"
    if cual in ("e1", "todos"):
        experimento_1()
    if cual in ("e2", "todos"):
        experimento_2()
    if cual in ("e3", "todos"):
        experimento_3()
