# Guía para la sustentación

## Orden sugerido (unos 15 minutos)

1. **Problema** (1 min): qué es el SCP y un ejemplo real, como ubicar estaciones de bomberos.
2. **Modelo** (2 min): variable x_j, función objetivo, restricción de cobertura y el ejemplo de 4 elementos del informe.
3. **Método exacto** (2 min): PuLP + CBC, ramificación y acotamiento, relajación lineal, cortes.
4. **Algoritmo genético** (4 min): cromosoma, aptitud, reparación, torneo, fusión, mutación, reemplazo y parada.
5. **Demostración en la interfaz** (3 min):
   - scp41: los dos métodos llegan al óptimo o muy cerca.
   - scp49: el AG no llega al óptimo; mostrar la curva de convergencia.
6. **Resultados y comparación** (3 min): tabla E1, figura de escalabilidad y tabla de ventajas y desventajas.

## Preguntas probables y respuestas cortas

**¿Por qué el SCP es difícil?**
Es NP-difícil (Karp, 1972). No se conoce un algoritmo polinomial, así que en el peor caso el
tiempo del método exacto crece de forma exponencial con el tamaño.

**¿Cuántas variables y restricciones tiene el modelo?**
n variables binarias y m restricciones de cobertura. En scp41 son 1000 y 200; en scp51, 2000 y 200.

**¿Qué hace CBC?**
Resuelve la relajación lineal, que da una cota inferior. Si hay variables fraccionarias, ramifica
(x_j = 0 / x_j = 1) y agrega cortes. Cuando la cota inferior coincide con la mejor solución entera,
el óptimo queda probado.

**¿Por qué es tan rápido en la OR-Library?**
La relajación lineal de estas instancias es muy ajustada y los cortes cierran la brecha casi sin
ramificar. En 600×6000 ya tarda 121 s, y en 800×8000 no termina en 180 s.

**¿Por qué todos los individuos son factibles?**
Por el operador de reparación. Agrega la columna con menor costo por fila nueva cubierta para cada
fila descubierta y luego quita las columnas redundantes, empezando por la más cara.

**¿Para qué sirve quitar las columnas redundantes?**
Si una columna se puede quitar sin descubrir ninguna fila, solo suma costo. Al quitarla, la
solución queda mínima y mejora sin perder factibilidad.

**¿Cómo funciona el cruce por fusión?**
Si los dos padres tienen el mismo gen, el hijo lo hereda. Si difieren, toma el gen del padre 1 con
probabilidad f2/(f1+f2), así que el padre más barato pesa más.

**¿Por qué la mutación es variable?**
Al principio la población es diversa y basta con poca mutación. Cuando converge, la mutación
aumenta hasta mf = 10 bits para reintroducir diversidad.

**¿Por qué se rechazan los duplicados?**
Para que la población no se llene de copias. Aun así, el 56 % de los hijos fueron duplicados, lo
que muestra que el AG converge rápido.

**¿El AG encontró el óptimo?**
Sí en 15 de 18 instancias, en 77 de 180 corridas (43 %). Las excepciones fueron scp49, scp52 y
scp53, donde la mejor corrida quedó a 3, 3 y 2 unidades del óptimo.

**¿Por qué el gap sale negativo en las instancias grandes?**
Porque CBC se detuvo por el límite de tiempo con una solución no óptima. El AG encontró una
solución mejor que esa en menos del 8 % del tiempo.

**¿Cómo se eligieron los parámetros?**
Se partió de Beasley y Chu (N = 100, mf = 10, mc = 200, mg = 2) y se validaron con el experimento
E3. Una población más grande o una mutación más fuerte bajan un poco el gap, pero las diferencias
son menores que la variación entre semillas.

**¿Cuál método es mejor?**
Depende del tamaño. Para instancias pequeñas y medianas, el exacto: es más rápido y garantiza el
óptimo. Para instancias grandes o con poco tiempo, el AG: escala casi de forma lineal y da buenas
soluciones.

**¿Qué mejorarían?**
Agregar búsqueda local (algoritmo memético), reiniciar parte de la población al estancarse, o usar
la solución del AG como punto de partida del solver exacto.

## Archivos clave del código, por si piden mostrarlos

- Modelo exacto: `src/exacto.py`, función `resolver_exacto` (unas 20 líneas).
- Reparación: `src/ga.py`, método `reparar`.
- Cruce: `cruce_fusion`. Mutación: `bits_a_mutar` y `mutar`.
- Ciclo principal: `ejecutar` (selección, cruce, mutación, reparación, reemplazo y parada).
