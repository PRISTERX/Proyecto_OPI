# Proyecto SET COVER: método exacto vs. algoritmo genético

Comparación entre la solución exacta (programación lineal entera con PuLP + CBC) y un
algoritmo genético (basado en Beasley y Chu, 1996) para el problema de cubrimiento de
conjuntos, usando las instancias de la OR-Library.

## Instalación

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
```

PuLP ya trae el solver CBC; no hay que instalar nada más. Se fija `pulp==2.9.0` porque la
versión 4 cambió la forma de crear variables.

## Uso

**Interfaz visual** (desde la carpeta del proyecto):

```bash
streamlit run app.py
```

Se abre en el navegador. Escoja la instancia en la barra lateral y pulse
"Resolver con el método exacto" y "Ejecutar el algoritmo genético"; la pestaña
"Comparación" muestra el gap, los tiempos y los gráficos.

**Desde la terminal** (desde la carpeta `src`):

```bash
python exacto.py ../datos/scp41.txt      # método exacto
python ga.py ../datos/scp41.txt          # algoritmo genético
python experimentos.py todos             # repite todos los experimentos (~30 min)
python figuras.py                        # regenera las figuras
python tablas.py                         # regenera las tablas LaTeX del informe
```

## Estructura

```
app.py                  Interfaz visual (Streamlit)
src/instancia.py        Lectura del formato OR-Library e instancias aleatorias
src/exacto.py           Modelo de programación lineal entera (PuLP + CBC)
src/ga.py               Algoritmo genético y heurística greedy de referencia
src/experimentos.py     Experimentos E1 (OR-Library), E2 (escalabilidad), E3 (parámetros)
src/figuras.py          Figuras del informe
src/tablas.py           Tablas LaTeX del informe
datos/                  Instancias scp41-scp49 y scp51-scp59 (OR-Library)
resultados/             CSV de los experimentos y figuras
informe/                Informe en LaTeX para Overleaf
```

## Resultados principales

| | Exacto (CBC) | Algoritmo genético | Greedy |
|---|---|---|---|
| Instancias con el óptimo (de 18) | 18 (probado) | 15 (mejor de 10 corridas) | 0 |
| Gap promedio | 0 % | 0,58 % | 4,3 % |
| Tiempo promedio | 0,09 s | 3,33 s | < 0,01 s |

En instancias aleatorias grandes (800×8000 y 1000×10000) CBC no termina en 180 s y el
AG encuentra soluciones iguales o mejores en unos 12-14 s.

## Instancias

Las instancias provienen de la OR-Library (Beasley, 1990):
http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
Las instancias scp410 y scp510 se pueden descargar de esa página y copiar en `datos/`;
el programa las detecta automáticamente.
