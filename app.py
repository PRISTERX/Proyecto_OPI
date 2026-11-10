"""
Interfaz visual del proyecto SET COVER: método exacto vs. algoritmo genético.

Ejecutar desde la carpeta del proyecto:
    streamlit run app.py
"""
import sys
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
sys.path.append(str(BASE / "src"))

from instancia import (leer_orlib, generar_aleatoria, instancias_disponibles,  # noqa: E402
                       OPTIMOS_LITERATURA)
from exacto import resolver_exacto  # noqa: E402
from ga import AlgoritmoGenetico, ParametrosAG, greedy_chvatal  # noqa: E402

COLOR = {"Exacto (PuLP + CBC)": "#2a78d6", "Algoritmo Genético": "#eb6834",
         "Greedy (Chvátal)": "#1baf7a"}
ESCALA = alt.Scale(domain=list(COLOR), range=list(COLOR.values()))

st.set_page_config(page_title="Set Cover: Exacto vs. AG", layout="wide")
st.title("Problema SET COVER: método exacto vs. algoritmo genético")
st.caption("Modelo exacto con PuLP + CBC  ·  Algoritmo genético basado en Beasley y Chu (1996)  ·  "
           "Instancias OR-Library")


# ---------------------------------------------------------------------------
# Barra lateral: instancia y parámetros
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_orlib(ruta: str):
    return leer_orlib(ruta)


@st.cache_data(show_spinner=False)
def cargar_aleatoria(m, n, dens, semilla):
    return generar_aleatoria(m, n, dens, 100, semilla)


with st.sidebar:
    st.header("1. Instancia")
    origen = st.radio("Origen", ["OR-Library", "Aleatoria", "Subir archivo"], horizontal=True)
    if origen == "OR-Library":
        archivos = instancias_disponibles(BASE / "datos")
        nombre = st.selectbox("Archivo", [a.stem for a in archivos])
        inst = cargar_orlib(str(BASE / "datos" / f"{nombre}.txt"))
    elif origen == "Aleatoria":
        m = st.number_input("Filas (m)", 10, 2000, 200, step=50)
        n = st.number_input("Columnas (n)", 20, 20000, 2000, step=500)
        dens = st.slider("Densidad", 0.01, 0.20, 0.02, 0.01)
        sem_inst = st.number_input("Semilla de la instancia", 0, 9999, 2026)
        inst = cargar_aleatoria(int(m), int(n), float(dens), int(sem_inst))
    else:
        archivo = st.file_uploader("Archivo .txt en formato OR-Library", type="txt")
        if archivo is None:
            st.info("Suba un archivo para continuar.")
            st.stop()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt",
                                         prefix=Path(archivo.name).stem + "_") as tmp:
            tmp.write(archivo.getvalue())
        inst = leer_orlib(tmp.name)
        inst.nombre = Path(archivo.name).stem

    st.header("2. Método exacto")
    lim_exacto = st.number_input("Límite de tiempo CBC (s)", 5, 3600, 300, step=5)

    st.header("3. Algoritmo genético")
    pob = st.number_input("Tamaño de población", 10, 500, 100, step=10)
    max_hijos = st.number_input("Máximo de hijos", 1000, 500000, 50000, step=5000)
    max_sin_mejora = st.number_input("Hijos sin mejora (parada)", 500, 200000, 15000, step=1000)
    tiempo_max = st.number_input("Tiempo máximo (s)", 5, 3600, 120, step=5)
    with st.expander("Mutación variable"):
        mf = st.number_input("mf (bits finales)", 1.0, 100.0, 10.0)
        mc = st.number_input("mc (hijos a mitad)", 1.0, 10000.0, 200.0)
        mg = st.number_input("mg (pendiente)", 0.1, 20.0, 2.0)
    semilla = st.number_input("Semilla del AG", 0, 99999, 1)
    corridas = st.number_input("Número de corridas", 1, 30, 1)

if st.session_state.get("instancia_actual") != inst.nombre:
    st.session_state.instancia_actual = inst.nombre
    for k in ("exacto", "ag", "greedy"):
        st.session_state.pop(k, None)

tab_inst, tab_ex, tab_ag, tab_comp, tab_exp = st.tabs(
    ["Instancia", "Método exacto", "Algoritmo genético", "Comparación", "Experimentos"])

# ---------------------------------------------------------------------------
# Pestaña: instancia
# ---------------------------------------------------------------------------
with tab_inst:
    res = inst.resumen()
    c = st.columns(5)
    c[0].metric("Filas (m)", res["Filas (m)"])
    c[1].metric("Columnas (n)", res["Columnas (n)"])
    c[2].metric("Densidad", f"{res['Densidad (%)']} %")
    c[3].metric("Costos", f"{res['Costo min']} – {res['Costo max']}")
    opt = OPTIMOS_LITERATURA.get(inst.nombre)
    c[4].metric("Óptimo en la literatura", opt if opt else "—")

    st.markdown(
        r"""
**Modelo matemático**

$$\min Z=\sum_{j=1}^{n} c_j x_j \quad \text{s.a.}\quad \sum_{j=1}^{n} a_{ij}x_j \ge 1 \;\; \forall i=1,\dots,m,
\qquad x_j\in\{0,1\}\;\; \forall j$$

donde $x_j=1$ si se selecciona el subconjunto (columna) $j$, $c_j$ es su costo y $a_{ij}=1$ si la columna $j$ cubre la fila $i$.
""")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Columnas que cubren cada fila**")
        df_f = pd.DataFrame({"Columnas que cubren la fila": inst.A.sum(axis=1)})
        st.altair_chart(alt.Chart(df_f).mark_bar(color="#2a78d6", cornerRadiusTopLeft=3,
                                                 cornerRadiusTopRight=3).encode(
            alt.X("Columnas que cubren la fila:Q", bin=alt.Bin(maxbins=30)),
            alt.Y("count():Q", title="Número de filas"), tooltip=["count():Q"]),
            width="stretch")
    with col2:
        st.markdown("**Distribución de costos**")
        df_c = pd.DataFrame({"Costo": inst.costos})
        st.altair_chart(alt.Chart(df_c).mark_bar(color="#2a78d6", cornerRadiusTopLeft=3,
                                                 cornerRadiusTopRight=3).encode(
            alt.X("Costo:Q", bin=alt.Bin(maxbins=30)),
            alt.Y("count():Q", title="Número de columnas"), tooltip=["count():Q"]),
            width="stretch")

# ---------------------------------------------------------------------------
# Pestaña: método exacto
# ---------------------------------------------------------------------------
with tab_ex:
    st.write("Resuelve el modelo de programación lineal entera con **CBC** (ramificación y corte).")
    if st.button("Resolver con el método exacto", type="primary"):
        with st.spinner("Resolviendo con CBC..."):
            st.session_state.exacto = resolver_exacto(inst, limite_tiempo=float(lim_exacto))
            st.session_state.greedy = greedy_chvatal(inst)
    r = st.session_state.get("exacto")
    if r:
        c = st.columns(5)
        c[0].metric("Costo (Z*)", f"{r['costo']:.0f}")
        c[1].metric("Columnas seleccionadas", r["num_columnas"])
        c[2].metric("Tiempo total", f"{r['tiempo_total']:.2f} s")
        c[3].metric("Estado", "Óptimo probado" if r["optimo_probado"] else "Límite de tiempo")
        c[4].metric("Factible", "Sí" if r["factible"] else "No")
        st.caption(f"Variables: {r['num_variables']}  ·  Restricciones: {r['num_restricciones']}  ·  "
                   f"Construcción del modelo: {r['tiempo_construccion']:.2f} s  ·  "
                   f"Solución CBC: {r['tiempo_solucion']:.2f} s")
        st.markdown("**Columnas seleccionadas (x_j = 1)**")
        st.code(", ".join(map(str, r["columnas"])))

# ---------------------------------------------------------------------------
# Pestaña: algoritmo genético
# ---------------------------------------------------------------------------
with tab_ag:
    st.write("Codificación binaria, torneo binario, cruce por fusión, mutación variable, "
             "reparación y reemplazo de estado estacionario.")
    if st.button("Ejecutar el algoritmo genético", type="primary"):
        resultados = []
        barra = st.progress(0.0, text="Ejecutando...")
        grafico = st.empty()
        for k in range(int(corridas)):
            params = ParametrosAG(tam_poblacion=int(pob), max_hijos=int(max_hijos),
                                  max_sin_mejora=int(max_sin_mejora), tiempo_max=float(tiempo_max),
                                  mf=mf, mc=mc, mg=mg, semilla=int(semilla) + k)
            puntos = []

            def avance(t, mejor, prom, k=k):
                puntos.append((t, mejor, prom))
                if t % 1000 == 0:
                    barra.progress(min(1.0, (k + t / params.max_hijos) / corridas),
                                   text=f"Corrida {k + 1}/{corridas} · hijo {t} · mejor = {mejor:.0f}")
                    d = pd.DataFrame(puntos, columns=["Hijos", "Mejor", "Promedio"])
                    grafico.line_chart(d.set_index("Hijos"))

            resultados.append(AlgoritmoGenetico(inst, params).ejecutar(callback=avance))
        barra.empty(); grafico.empty()
        st.session_state.ag = resultados
        if "greedy" not in st.session_state:
            st.session_state.greedy = greedy_chvatal(inst)

    res_ag = st.session_state.get("ag")
    if res_ag:
        mejor = min(res_ag, key=lambda r: r["costo"])
        costos = [r["costo"] for r in res_ag]
        c = st.columns(5)
        c[0].metric("Mejor costo", f"{mejor['costo']:.0f}")
        c[1].metric("Costo promedio", f"{np.mean(costos):.1f}")
        c[2].metric("Tiempo promedio", f"{np.mean([r['tiempo_total'] for r in res_ag]):.2f} s")
        c[3].metric("Hijos (mejor corrida)", mejor["hijos_generados"])
        c[4].metric("Parada", mejor["motivo_parada"])

        filas = []
        for k, r in enumerate(res_ag):
            h = r["historial"]
            filas += [{"Corrida": f"Semilla {r['parametros']['semilla']}", "Hijos": int(a),
                       "Serie": "Mejor", "Costo": b} for a, b, _, _ in h]
            filas += [{"Corrida": f"Semilla {r['parametros']['semilla']}", "Hijos": int(a),
                       "Serie": "Promedio población", "Costo": cc} for a, _, cc, _ in h]
        dfh = pd.DataFrame(filas)
        st.markdown("**Convergencia** (mejor individuo y promedio de la población)")
        base = alt.Chart(dfh).encode(
            x=alt.X("Hijos:Q", title="Hijos generados"),
            y=alt.Y("Costo:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Serie:N", scale=alt.Scale(domain=["Mejor", "Promedio población"],
                                                       range=["#eb6834", "#a3a29c"])),
            detail="Corrida:N", tooltip=["Corrida", "Hijos", "Serie", "Costo"])
        capa = base.mark_line(strokeWidth=2)
        if st.session_state.get("exacto"):
            regla = alt.Chart(pd.DataFrame({"Óptimo": [st.session_state.exacto["costo"]]})).mark_rule(
                color="#2a78d6", strokeDash=[6, 4], strokeWidth=2).encode(y="Óptimo:Q",
                                                                         tooltip=["Óptimo:Q"])
            capa = capa + regla
        st.altair_chart(capa.interactive(), width="stretch")
        st.caption("La línea azul punteada es el óptimo del método exacto (si ya se ejecutó).")

        st.markdown("**Detalle por corrida**")
        st.dataframe(pd.DataFrame([{
            "Semilla": r["parametros"]["semilla"], "Costo": r["costo"],
            "Columnas": r["num_columnas"], "Tiempo (s)": round(r["tiempo_total"], 2),
            "Hijo de la mejor solución": r["hijo_mejor"], "Hijos totales": r["hijos_generados"],
            "Duplicados descartados": r["duplicados"], "Parada": r["motivo_parada"],
            "Factible": r["factible"]} for r in res_ag]), hide_index=True,
            width="stretch")
        st.markdown("**Columnas seleccionadas por la mejor corrida**")
        st.code(", ".join(map(str, mejor["columnas"])))

# ---------------------------------------------------------------------------
# Pestaña: comparación
# ---------------------------------------------------------------------------
with tab_comp:
    ex, ag, gr = (st.session_state.get(k) for k in ("exacto", "ag", "greedy"))
    if not (ex and ag):
        st.info("Ejecute el método exacto y el algoritmo genético para ver la comparación.")
    else:
        mejor = min(ag, key=lambda r: r["costo"])
        prom = float(np.mean([r["costo"] for r in ag]))
        t_ag = float(np.mean([r["tiempo_total"] for r in ag]))
        gap_mejor = 100 * (mejor["costo"] - ex["costo"]) / ex["costo"]
        gap_prom = 100 * (prom - ex["costo"]) / ex["costo"]

        c = st.columns(4)
        c[0].metric("Gap del mejor AG", f"{gap_mejor:.2f} %")
        c[1].metric("Gap promedio AG", f"{gap_prom:.2f} %")
        c[2].metric("Tiempo exacto", f"{ex['tiempo_total']:.2f} s")
        c[3].metric("Tiempo AG (prom.)", f"{t_ag:.2f} s")

        if gap_mejor <= 1e-9:
            st.success("El algoritmo genético encontró la solución óptima.")
        else:
            st.warning(f"El algoritmo genético quedó a {mejor['costo'] - ex['costo']:.0f} unidades "
                       f"({gap_mejor:.2f} %) del óptimo.")

        tabla = pd.DataFrame([
            {"Método": "Exacto (PuLP + CBC)", "Costo": ex["costo"], "Columnas": ex["num_columnas"],
             "Tiempo (s)": ex["tiempo_total"], "Gap (%)": 0.0,
             "Optimalidad garantizada": "Sí" if ex["optimo_probado"] else "No"},
            {"Método": "Algoritmo Genético", "Costo": mejor["costo"], "Columnas": mejor["num_columnas"],
             "Tiempo (s)": t_ag, "Gap (%)": gap_mejor, "Optimalidad garantizada": "No"},
            {"Método": "Greedy (Chvátal)", "Costo": gr["costo"], "Columnas": int(gr["x"].sum()),
             "Tiempo (s)": gr["tiempo_total"],
             "Gap (%)": 100 * (gr["costo"] - ex["costo"]) / ex["costo"],
             "Optimalidad garantizada": "No"},
        ])
        st.dataframe(tabla.round(3), hide_index=True, width="stretch")

        g1, g2 = st.columns(2)
        for col, campo, titulo in ((g1, "Costo", "Costo de la solución"),
                                   (g2, "Tiempo (s)", "Tiempo de cómputo (s)")):
            with col:
                st.markdown(f"**{titulo}**")
                ch = alt.Chart(tabla).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                                               size=48).encode(
                    x=alt.X("Método:N", sort=list(COLOR), title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(f"{campo}:Q", title=campo),
                    color=alt.Color("Método:N", scale=ESCALA, legend=None),
                    tooltip=["Método", alt.Tooltip(f"{campo}:Q", format=".3f")])
                txt = ch.mark_text(dy=-8, color="#52514e").encode(text=alt.Text(f"{campo}:Q",
                                                                               format=".2f"))
                st.altair_chart(ch + txt, width="stretch")

        comunes = set(ex["columnas"]) & set(mejor["columnas"])
        st.markdown(f"**Coincidencia de columnas:** {len(comunes)} columnas aparecen en ambas "
                    f"soluciones (exacto: {ex['num_columnas']}, AG: {mejor['num_columnas']}).")

        salida = pd.DataFrame({
            "columna_j": np.arange(1, inst.n + 1), "costo_cj": inst.costos,
            "x_exacto": ex["x"], "x_AG": mejor["x"], "x_greedy": gr["x"]})
        st.download_button("Descargar soluciones (CSV)", salida.to_csv(index=False).encode(),
                           file_name=f"soluciones_{inst.nombre}.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Pestaña: experimentos precalculados
# ---------------------------------------------------------------------------
with tab_exp:
    ruta = BASE / "resultados" / "e1_resumen.csv"
    if not ruta.exists():
        st.info("Ejecute `python src/experimentos.py` para generar los resultados.")
    else:
        st.markdown("**E1. Instancias OR-Library: exacto vs. AG (10 corridas) vs. greedy**")
        df = pd.read_csv(ruta)
        cols = ["instancia", "m", "n", "costo_exacto", "tiempo_exacto", "costo_greedy",
                "ag_mejor", "ag_promedio", "ag_peor", "gap_promedio", "tiempo_ag_prom", "exitos"]
        st.dataframe(df[cols].round(2), hide_index=True, width="stretch")
        r2 = BASE / "resultados" / "e2_escalabilidad.csv"
        if r2.exists():
            st.markdown("**E2. Escalabilidad en instancias aleatorias**")
            st.dataframe(pd.read_csv(r2).round(2), hide_index=True, width="stretch")
        r3 = BASE / "resultados" / "e3_sensibilidad.csv"
        if r3.exists():
            st.markdown("**E3. Sensibilidad de parámetros (gap promedio %)**")
            d3 = pd.read_csv(r3).groupby(["instancia", "poblacion", "mf"])[["gap", "tiempo"]] \
                .mean().round(2).reset_index()
            st.dataframe(d3, hide_index=True, width="stretch")
        figs = sorted((BASE / "resultados" / "figuras").glob("*.png"))
        for f in figs:
            st.image(str(f), caption=f.stem, width="stretch")
