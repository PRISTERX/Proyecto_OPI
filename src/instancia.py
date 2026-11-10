"""
Lectura y generación de instancias del problema SET COVER (SCP).

Formato OR-Library (Beasley, 1990):
    m n
    c_1 c_2 ... c_n                      (costo de cada columna)
    para cada fila i = 1..m:
        k_i  j_1 j_2 ... j_k             (cuántas columnas cubren la fila i y cuáles son)

En este proyecto:
    - filas    = elementos que deben quedar cubiertos (i = 1..m)
    - columnas = subconjuntos disponibles (j = 1..n), cada uno con costo c_j
    - A[i, j]  = 1 si el subconjunto j cubre el elemento i
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class InstanciaSCP:
    nombre: str
    m: int                      # número de filas (elementos)
    n: int                      # número de columnas (subconjuntos)
    costos: np.ndarray          # vector c de tamaño n
    A: np.ndarray               # matriz de cobertura m x n (booleana)
    filas_de_col: list = field(default_factory=list)   # filas que cubre cada columna
    cols_de_fila: list = field(default_factory=list)   # columnas que cubren cada fila

    def __post_init__(self):
        if not self.filas_de_col:
            self.filas_de_col = [np.flatnonzero(self.A[:, j]) for j in range(self.n)]
        if not self.cols_de_fila:
            self.cols_de_fila = [np.flatnonzero(self.A[i, :]) for i in range(self.m)]

    @property
    def densidad(self) -> float:
        """Porcentaje de unos en la matriz A."""
        return 100.0 * self.A.sum() / (self.m * self.n)

    def es_factible(self, x: np.ndarray) -> bool:
        """Una solución es factible si cada fila queda cubierta al menos una vez."""
        return bool(np.all(self.A[:, x.astype(bool)].any(axis=1)))

    def costo(self, x: np.ndarray) -> float:
        return float(self.costos @ x)

    def resumen(self) -> dict:
        return {
            "Instancia": self.nombre,
            "Filas (m)": self.m,
            "Columnas (n)": self.n,
            "Densidad (%)": round(self.densidad, 2),
            "Costo min": int(self.costos.min()),
            "Costo max": int(self.costos.max()),
        }


def leer_orlib(ruta) -> InstanciaSCP:
    """Lee un archivo de set covering en formato OR-Library."""
    ruta = Path(ruta)
    numeros = [int(t) for t in ruta.read_text().split()]
    pos = 0
    m, n = numeros[pos], numeros[pos + 1]
    pos += 2
    costos = np.array(numeros[pos:pos + n], dtype=float)
    pos += n
    A = np.zeros((m, n), dtype=bool)
    for i in range(m):
        k = numeros[pos]
        pos += 1
        columnas = np.array(numeros[pos:pos + k]) - 1      # el archivo usa índices desde 1
        pos += k
        A[i, columnas] = True
    return InstanciaSCP(ruta.stem, m, n, costos, A)


def generar_aleatoria(m: int, n: int, densidad: float = 0.05,
                      costo_max: int = 100, semilla: int = 0,
                      nombre: str | None = None) -> InstanciaSCP:
    """
    Genera una instancia aleatoria factible (cada fila es cubierta por
    al menos dos columnas), útil para pruebas pequeñas en la interfaz.
    """
    rng = np.random.default_rng(semilla)
    A = rng.random((m, n)) < densidad
    for i in range(m):                      # garantiza al menos 2 columnas por fila
        faltan = 2 - A[i].sum()
        if faltan > 0:
            A[i, rng.choice(n, size=faltan, replace=False)] = True
    for j in range(n):                      # cada columna cubre al menos una fila
        if not A[:, j].any():
            A[rng.integers(m), j] = True
    costos = rng.integers(1, costo_max + 1, size=n).astype(float)
    nombre = nombre or f"rand_{m}x{n}_s{semilla}"
    return InstanciaSCP(nombre, m, n, costos, A)


def instancias_disponibles(carpeta) -> list[Path]:
    """Lista los archivos scp*.txt ordenados (scp41, scp42, ..., scp59)."""
    archivos = list(Path(carpeta).glob("scp*.txt"))

    def clave(p: Path):
        nombre = p.stem.replace("scp", "")
        return (int(nombre[0]), int(nombre[1:]))

    return sorted(archivos, key=clave)


# Valores óptimos reportados en la literatura (Beasley y Chu, 1996)
OPTIMOS_LITERATURA = {
    "scp41": 429, "scp42": 512, "scp43": 516, "scp44": 494, "scp45": 512,
    "scp46": 560, "scp47": 430, "scp48": 492, "scp49": 641, "scp410": 514,
    "scp51": 253, "scp52": 302, "scp53": 226, "scp54": 242, "scp55": 211,
    "scp56": 213, "scp57": 293, "scp58": 288, "scp59": 279, "scp510": 265,
}
