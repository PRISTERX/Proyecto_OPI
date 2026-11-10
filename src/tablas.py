"""Genera las tablas LaTeX del informe a partir de ../resultados/*.csv"""
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "resultados"
SAL = BASE / "informe" / "tablas"
SAL.mkdir(parents=True, exist_ok=True)


def escribir(nombre, filas):
    """Escribe la tabla usando coma decimal (formato en español)."""
    texto = "\n".join(filas + ["\\bottomrule"])
    texto = re.sub(r"(\d)\.(\d)", r"\1,\2", texto)
    texto = re.sub(r"(?<![\w$])-(\d)", r"$-$\1", texto)
    (SAL / nombre).write_text(texto)


def tabla_e1():
    d = pd.read_csv(RES / "e1_resumen.csv")
    filas = []
    for _, r in d.iterrows():
        gap_gr = 100 * (r.costo_greedy - r.costo_exacto) / r.costo_exacto
        filas.append(
            f"{r.instancia} & {r.m}$\\times${r.n} & {r.costo_exacto:.0f} & {r.tiempo_exacto:.2f} & "
            f"{r.costo_greedy:.0f} ({gap_gr:.1f}) & {r.ag_mejor:.0f} & {r.ag_promedio:.1f} & "
            f"{r.ag_peor:.0f} & {r.gap_promedio:.2f} & {r.tiempo_ag_prom:.2f} & {r.exitos}/10 \\\\")
    c = pd.read_csv(RES / "e1_ag_corridas.csv")
    gap_gr_prom = (100 * (d.costo_greedy - d.costo_exacto) / d.costo_exacto).mean()
    filas.append("\\midrule")
    filas.append(f"\\textbf{{Promedio}} & & & {d.tiempo_exacto.mean():.2f} & ({gap_gr_prom:.1f}) & & & & "
                 f"{c.gap.mean():.2f} & {c.tiempo_ag.mean():.2f} & {d.exitos.sum()}/180 \\\\")
    escribir("tabla_e1.tex", filas)


def tabla_e2():
    d = pd.read_csv(RES / "e2_escalabilidad.csv")
    filas = []
    for _, r in d.iterrows():
        estado = "Óptimo probado" if r.optimo_probado else "Límite 180 s"
        filas.append(
            f"{r.m}$\\times${r.n} & {r.costo_exacto:.0f} & {estado} & {r.tiempo_exacto:.2f} & "
            f"{r.ag_mejor:.0f} & {r.ag_promedio:.2f} & {r.tiempo_ag_prom:.2f} & {r.gap_promedio:.2f} \\\\")
    escribir("tabla_e2.tex", filas)


def tabla_e3():
    d = pd.read_csv(RES / "e3_sensibilidad.csv")
    g = d.groupby(["poblacion", "mf", "instancia"]).agg(gap=("gap", "mean"),
                                                         t=("tiempo", "mean")).unstack("instancia")
    filas = []
    for (pob, mf), r in g.iterrows():
        filas.append(f"{pob} & {mf} & {r[('gap', 'scp49')]:.2f} & {r[('t', 'scp49')]:.2f} & "
                     f"{r[('gap', 'scp57')]:.2f} & {r[('t', 'scp57')]:.2f} \\\\")
    escribir("tabla_e3.tex", filas)


if __name__ == "__main__":
    for f in (tabla_e1, tabla_e2, tabla_e3):
        try:
            f(); print("ok", f.__name__)
        except FileNotFoundError as e:
            print("falta", e)
