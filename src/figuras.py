"""Genera las figuras del informe a partir de ../resultados/*.csv"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "resultados"
FIG = RES / "figuras"
FIG.mkdir(parents=True, exist_ok=True)

AZUL, NARANJA, AQUA, GRIS, TINTA, TINTA2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c9c8c2",
    "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": TINTA,
})


def guardar(fig, nombre):
    fig.savefig(FIG / f"{nombre}.png")
    fig.savefig(FIG / f"{nombre}.pdf")
    plt.close(fig)


def fig_gap():
    df = pd.read_csv(RES / "e1_resumen.csv")
    corr = pd.read_csv(RES / "e1_ag_corridas.csv")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    g_min = corr.groupby("instancia", sort=False)["gap"].min().values
    g_max = corr.groupby("instancia", sort=False)["gap"].max().values
    ax.vlines(x, g_min, g_max, color=NARANJA, lw=2, alpha=0.45, label="AG: rango (10 corridas)")
    ax.plot(x, df["gap_promedio"], "o", color=NARANJA, ms=6, mec="white", mew=1.5,
            label="AG: gap promedio")
    gap_gr = 100 * (df["costo_greedy"] - df["costo_exacto"]) / df["costo_exacto"]
    ax.plot(x, gap_gr, "D", color=AQUA, ms=5, mec="white", mew=1.2, label="Greedy (Chvátal)")
    ax.axhline(0, color=AZUL, lw=2, label="Exacto (óptimo, gap 0 %)")
    ax.set_xticks(x, df["instancia"], rotation=45, ha="right")
    ax.set_ylabel("Gap respecto al óptimo (%)")
    ax.set_title("Calidad de la solución por instancia")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8)
    guardar(fig, "fig_gap_instancias")


def fig_tiempos():
    df = pd.read_csv(RES / "e1_resumen.csv")
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar(x - w / 2 - 0.01, df["tiempo_exacto"], w, color=AZUL, label="Exacto (CBC)")
    ax.bar(x + w / 2 + 0.01, df["tiempo_ag_prom"], w, color=NARANJA, label="AG (promedio total)")
    ax.plot(x + w / 2 + 0.01, df["tiempo_mejor_prom"], "_", color=TINTA, ms=9, mew=2,
            label="AG: tiempo hasta su mejor solución")
    ax.set_yscale("log")
    ax.set_xticks(x, df["instancia"], rotation=45, ha="right")
    ax.set_ylabel("Tiempo (s, escala log)")
    ax.set_title("Tiempo de cómputo por instancia")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8)
    guardar(fig, "fig_tiempos_instancias")


def fig_convergencia():
    hist = json.loads((RES / "e1_historiales.json").read_text())
    ex = pd.read_csv(RES / "e1_exacto.csv").set_index("instancia")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, nombre in zip(axes, ["scp41", "scp49"]):
        opt = ex.loc[nombre, "costo_exacto"]
        for k in range(1, 11):
            h = np.array(hist[f"{nombre}_s{k}"])
            ax.plot(h[:, 0], h[:, 1], color=NARANJA, lw=1.2, alpha=0.55,
                    label="Mejor individuo (10 semillas)" if k == 1 else None)
        h = np.array(hist[f"{nombre}_s1"])
        ax.plot(h[:, 0], h[:, 2], color=GRIS, lw=1.5, label="Promedio de la población (semilla 1)")
        ax.axhline(opt, color=AZUL, lw=1.8, ls="--", label=f"Óptimo exacto")
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(0, 4e4)
        ax.set_ylim(opt * 0.99, opt * 1.12)
        ax.set_title(f"{nombre} (óptimo = {opt:.0f})")
        ax.set_xlabel("Hijos generados (escala log)")
    axes[0].set_ylabel("Costo")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.25), ncol=3, fontsize=8)
    guardar(fig, "fig_convergencia")


def fig_escalabilidad():
    df = pd.read_csv(RES / "e2_escalabilidad.csv")
    etiquetas = [f"{m}×{n}" for m, n in zip(df["m"], df["n"])]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    ax.plot(x, df["tiempo_exacto"], "-o", color=AZUL, lw=2, ms=6, mec="white", mew=1.5,
            label="Exacto (CBC)")
    ax.plot(x, df["tiempo_ag_prom"], "-s", color=NARANJA, lw=2, ms=6, mec="white", mew=1.5,
            label="AG (promedio)")
    sin_prueba = x[~df["optimo_probado"].values]
    if len(sin_prueba):
        ax.annotate("límite de 180 s\n(sin probar optimalidad)", (sin_prueba.mean(), 180),
                    textcoords="offset points", xytext=(0, -24), ha="center", fontsize=7,
                    color=TINTA2)
    ax.set_yscale("log")
    ax.set_xticks(x, etiquetas, rotation=30)
    ax.set_ylabel("Tiempo (s, escala log)")
    ax.set_title("Tiempo vs. tamaño")
    ax.legend(fontsize=8)
    ax = axes[1]
    colores = [NARANJA if ok else "#f4b293" for ok in df["optimo_probado"]]
    ax.bar(x, df["gap_promedio"], 0.55, color=colores)
    for xi, g, ok in zip(x, df["gap_promedio"], df["optimo_probado"]):
        ax.text(xi, g, f"{g:.1f}%".replace(".", ",") + ("" if ok else "*"), ha="center",
                va="bottom" if g >= 0 else "top", fontsize=7.5, color=TINTA2)
    ax.axhline(0, color=TINTA2, lw=0.8)
    ax.set_ylim(min(df["gap_promedio"].min() - 0.5, -0.2), df["gap_promedio"].max() + 0.5)
    ax.set_xticks(x, etiquetas, rotation=30)
    ax.set_ylabel("Gap promedio del AG (%)")
    ax.set_title("Gap del AG vs. tamaño")
    guardar(fig, "fig_escalabilidad")


def fig_sensibilidad():
    df = pd.read_csv(RES / "e3_sensibilidad.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.25})
    for ax, nombre in zip(axes, ["scp49", "scp57"]):
        tabla = df[df["instancia"] == nombre].pivot_table(index="poblacion", columns="mf",
                                                           values="gap", aggfunc="mean")
        im = ax.imshow(tabla.values, cmap="Oranges", vmin=0, vmax=max(df["gap"].max() * 0.8, 1))
        for i in range(tabla.shape[0]):
            for j in range(tabla.shape[1]):
                v = tabla.values[i, j]
                ax.text(j, i, f"{v:.2f}%".replace(".", ","), ha="center", va="center", fontsize=8,
                        color="white" if v > df["gap"].max() * 0.45 else TINTA)
        ax.set_xticks(range(tabla.shape[1]), [f"mf={c}" for c in tabla.columns])
        ax.set_yticks(range(tabla.shape[0]), [f"N={r}" for r in tabla.index])
        ax.grid(False)
        ax.set_title(f"{nombre}")
        if nombre == "scp57":
            ax.tick_params(labelleft=False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Gap promedio, 5 semillas (%)")
    guardar(fig, "fig_sensibilidad")


def fig_mutacion():
    mf, mc, mg = 10, 200, 2
    t = np.arange(0, 1001)
    k = [math.ceil(mf / (1 + math.exp(-4 * mg * (ti - mc) / mf))) for ti in t]
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.step(t, k, where="post", color=NARANJA, lw=2)
    ax.set_xlabel("Hijos generados t")
    ax.set_ylabel("Bits mutados")
    ax.set_title("Tasa de mutación variable")
    guardar(fig, "fig_mutacion")


if __name__ == "__main__":
    for f in (fig_mutacion, fig_gap, fig_tiempos, fig_convergencia, fig_escalabilidad,
              fig_sensibilidad):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("falta", f.__name__, e)
