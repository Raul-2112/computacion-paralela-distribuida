# lab_cuda_basic_intermediate | Valentina Cerpa & Raul Delgado

---

## Equipo

| | Colaborador | GitHub |
|---|---|---|
| 👤 | Valentina Cerpa | [@Valen187](https://github.com/Valen187) |
| 👤 | Raul Delgado | [@Raul-2112](https://github.com/Raul-2112) |


---
## Índice
1. [Ejercicio 1 — Hola GPU](#ejercicio-1)
2. [Ejercicio 2 — Copia de Matriz 2D](#ejercicio-2)
3. [Ejercicio 3 — Información del Device](#ejercicio-3)
4. [Ejercicio 4 — Suma de Vectores Paralela](#ejercicio-4)
5. [Ejercicio 5 — Cuadrado de Elementos In-place](#ejercicio-5)
6. [Ejercicio 6 — Kernel 2D: Inicialización de Matriz](#ejercicio-6)
7. [Ejercicio 7 — Reducción Paralela con Shared Memory](#ejercicio-7)
8. [Ejercicio 8 — Multiplicación Escalar y Medición de Tiempo](#ejercicio-8)
9. [Ejercicio 9 — Producto Punto de Vectores](#ejercicio-9)

---

## Ejercicio 1 — Hola GPU

### ¿Qué hace el código?
Inicializa un arreglo de 10 enteros en la CPU (valores `0, 3, 6, …, 27`), reserva memoria equivalente en la GPU con `cudaMalloc`, copia los datos `CPU → GPU` y luego `GPU → CPU` en un arreglo distinto. Finalmente compara elemento a elemento con `h_datos[i] != h_resultado[i]` para confirmar que todos los valores llegaron intactos.

### Comandos
```bash
nvcc ejercicio1_hola_gpu.cu -o ejercicio1
./ejercicio1
```

> 📸 **Evidencia:**
![Ejercicio 1](ejercicio1_hola_gpu/img/ejercicio1_hola_gpu.png)
---

## Ejercicio 2 — Copia de Matriz 2D

### ¿Qué hace el código?
Inicializa una matriz 3×4 de `float` aplanada en un arreglo 1D (valores `1.5, 3.0, 4.5, …`). La transfiere `CPU → GPU → CPU` sin ningún cálculo en la GPU. La **TAREA** añade un bucle de verificación que compara cada par `h_original[i]` vs `h_recuperada[i]` usando `fabsf(a - b) < 1e-5f` para tolerar errores de redondeo propios del tipo `float`.

### Comandos
```bash
nvcc ejercicio2_matriz.cu -o ejercicio2
./ejercicio2
```

### Salida esperada
```
Matriz original (CPU):
   1.5    3.0    4.5    6.0
   7.5    9.0   10.5   12.0
  13.5   15.0   16.5   18.0

[OK] Datos enviados a la GPU

Matriz recuperada desde GPU:
   1.5    3.0  ...

Verificacion automatica (tolerancia 1e-5):
  Todos los 12 elementos coinciden. [OK]
```

### TAREA — Verificación automática
Se usa `fabsf` en lugar de `==` porque los `float` tienen representación binaria limitada; una comparación exacta puede fallar incluso cuando el valor es "el mismo". La tolerancia `1e-5f` es segura para operaciones de copia pura donde no hay aritmética.

> 📸 **Evidencia:** 
![Ejercicio 2](ejercicio2_matriz/img/ejercicio2_matriz.png)
---

## Ejercicio 3 — Información del Device 

### ¿Qué hace el código?
Usa `cudaGetDeviceCount` para detectar cuántas GPUs hay disponibles y `cudaGetDeviceProperties` para leer sus propiedades: nombre, compute capability, memoria total, shared memory por bloque, máximo de hilos por bloque, número de SMs, frecuencia, ancho de bus y dimensiones máximas de grilla/bloque. La **TAREA** calcula el total de hilos simultáneos máximos.

### Comandos
```bash
nvcc ejercicio3_device_info.cu -o ejercicio3
./ejercicio3
```

### Salida esperada (GTX 1060)
```
GPUs CUDA disponibles en este sistema: 1

=== GPU 0: GeForce GTX 1060 ===
  Compute Capability     : 6.1
  Memoria Global         : 6.00 GB
  Memoria Compartida/Blq : 48 KB
  Hilos maximos/Bloque   : 1024
  Multiprocessors (SM)   : 10
  Hilos maximos/SM       : 2048
  Frecuencia del reloj   : 1.71 GHz
  Ancho de bus de memoria: 192 bits
  Dim. maxima de bloque  : (1024, 1024, 64)
  Dim. maxima de grilla  : (2147483647, 65535, 65535)

  >>> Hilos simultaneos maximos (SM x hilos/SM): 20480
      (10 SM  x  2048 hilos/SM)
```

### TAREA — Total de hilos simultáneos
La fórmula es:

```
Total hilos = multiProcessorCount × maxThreadsPerMultiProcessor
```

Para la GTX 1060: `10 × 2048 = 20 480 hilos simultáneos`. Este número representa la *ocupación máxima teórica*: cuántos hilos pueden estar residentes en la GPU al mismo tiempo (en distintos estados de ejecución). Es diferente del número total de hilos que puede *lanzar* una grilla, que es potencialmente miles de millones.

> 📸 **Evidencia:** 
![Ejercicio 3](ejercicio3_device_info/img/ejercicio3_device_info.png)
---

## Ejercicio 4 — Suma de Vectores Paralela 

### ¿Qué hace el código?
El "Hola Mundo" de CUDA. Crea dos vectores `h_A` y `h_B` de 1 000 000 de `float`, los copia a la GPU, lanza el kernel `sumaVectores` donde cada hilo calcula `d_C[idx] = d_A[idx] + d_B[idx]`, y verifica que el resultado sea `3.0f` en toda posición. El número de bloques se calcula con la fórmula estándar:

```c
int numBloques = (N + THREADS_POR_BLOQUE - 1) / THREADS_POR_BLOQUE;
// = (1 000 000 + 255) / 256 = 3907 bloques
```

### Comandos
```bash
nvcc ejercicio4_suma_vectores.cu -o ejercicio4
./ejercicio4
```

### Salida esperada
```
Lanzando 3907 bloques x 256 hilos = 1000192 hilos totales
h_C[0]   = 3.0 (esperado: 3.0)
h_C[N-1] = 3.0 (esperado: 3.0)

[OK] Suma de vectores completada.
```

**Nota:** Se lanzan 1 000 192 hilos aunque el vector tiene 1 000 000 elementos. El `if (idx < n)` en el kernel protege de escrituras fuera de límites para los 192 hilos sobrantes.

> 📸 **Evidencia:** 
![Ejercicio 4](ejercicio4_suma_vectores/img/ejercicio4_suma_vectores.png)
---

## Ejercicio 5 — Cuadrado de Elementos In-place 

### ¿Qué hace el código?
Inicializa `h_datos = [1, 2, 3, …, 20]`, lo copia a la GPU y lanza el kernel `cuadradoInPlace` que escribe `d_datos[idx] = d_datos[idx] * d_datos[idx]` sobre el **mismo** puntero (in-place). Recupera el resultado y la **TAREA** verifica que `h_datos[i] == (i+1)^2` para todo `i`.

### Comandos
```bash
nvcc ejercicio5_cuadrado.cu -o ejercicio5
./ejercicio5
```

### Salida esperada
```
Datos originales:
   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
Despues de elevar al cuadrado en GPU:
   1   4   9  16  25  36  49  64  81 100 121 144 169 196 225 256 289 324 361 400

Verificacion (esperado: 1, 4, 9, 16, ... 400):
  Todos los 20 elementos son correctos. [OK]
```

### TAREA — Verificación
El bucle compara `h_datos[i]` con `(i+1)*(i+1)`. Si alguno difiere, imprime el índice y los valores. La operación in-place es posible porque cada hilo escribe **solo** en su propia posición `idx`; no hay dependencias entre hilos.

> 📸 **Evidencia:** 
![Ejercicio 5](ejercicio5_cuadrado/img/ejercicio5_cuadrado.png)
---

## Ejercicio 6 — Kernel 2D: Inicialización de Matriz

### ¿Qué hace el código?
Lanza un kernel con dimensiones `dim3 hilosPorBloque(COLS, FILAS)` para que cada hilo corresponda a una celda `(fila, col)` de la matriz 4×5. El **kernel original** almacena el índice lineal `fila * COLS + col`. La **TAREA** implementa un segundo kernel que almacena `fila + col`.

### Comandos
```bash
nvcc ejercicio6_kernel2d.cu -o ejercicio6
./ejercicio6
```

### Salida esperada
```
Kernel 1 - mat[i][j] = indice lineal:
  0   1   2   3   4
  5   6   7   8   9
 10  11  12  13  14
 15  16  17  18  19

Kernel 2 (TAREA) - mat[i][j] = i + j:
  0   1   2   3   4
  1   2   3   4   5
  2   3   4   5   6
  3   4   5   6   7
```

### TAREA — Kernel con `mat[i][j] = i + j`
El único cambio respecto al kernel original es reemplazar `d_mat[idx] = idx` por `d_mat[idx] = fila + col`. Las variables `fila` y `col` ya estaban calculadas para el guard de límites, por lo que no se necesita lógica adicional.

> 📸 **Evidencia:** 
![Ejercicio 6](ejercicio6_kernel2d/img/ejercicio6_kernel2d.png)
---

## Ejercicio 7 — Reducción Paralela con Shared Memory 

### ¿Qué hace el código?
Suma 1024 enteros (todos `= 1`, resultado esperado `1024`) usando **reducción en árbol** con `shared memory`. El kernel carga los datos globales a `s_datos[]` (SRAM del bloque), luego en cada iteración la mitad activa de los hilos acumula su valor con el del hilo a distancia `stride`. Al final, el hilo 0 de cada bloque escribe la suma parcial en `d_salida[blockIdx.x]`. La CPU suma los 4 parciales.

### Comandos
```bash
nvcc ejercicio7_reduccion.cu -o ejercicio7
./ejercicio7
```

### Salida esperada
```
Suma esperada  (CPU): 1024
Bloques lanzados: 4  |  Hilos por bloque: 256
Suma calculada (GPU): 1024
[OK] Resultados identicos!
```

### Por qué `__syncthreads()` es crucial
Sin la barrera, los hilos del nivel siguiente del árbol podrían leer `s_datos[tid + stride]` **antes** de que el hilo vecino haya terminado de escribirlo. `__syncthreads()` garantiza una **barrera de memoria** interna al bloque: ningún hilo avanza hasta que todos llegaron a ese punto, eliminando la condición de carrera (*race condition*).

> 📸 **Evidencia:** 
![Ejercicio 7](ejercicio7_reduccion/img/ejercicio7_reduccion.png)
---

## Ejercicio 8 — Multiplicación Escalar y Medición de Tiempo

### ¿Qué hace el código?
Multiplica 10 000 000 de `float` por el escalar `2.5f`. Mide el tiempo GPU con **CUDA Events** (`cudaEventRecord` + `cudaEventElapsedTime`) y el tiempo CPU con `clock()`. Calcula el *bandwidth* efectivo de la GPU en GB/s y compara ambos tiempos.

### Comandos
```bash
nvcc ejercicio8_tiempo.cu -o ejercicio8
./ejercicio8
```

### Salida esperada (referencial, varía según hardware)
```
Tiempo CPU      : 28.4000 ms
h_cpu[0]        : 2.5 (esperado 2.5)

Tiempo GPU      : 1.2345 ms
Bandwidth GPU   : 64.85 GB/s
h_vec[0]        : 2.5 (esperado 2.5)

=== Comparacion de rendimiento ===
  CPU : 28.4000 ms
  GPU : 1.2345 ms
  La GPU es ~23.0x mas rapida que la CPU en esta operacion.
```

### TAREA — Comparación CPU vs GPU
La CPU realiza la operación secuencialmente, mientras la GPU lo hace con miles de hilos en paralelo. En operaciones *memory-bound* como esta (sin lógica compleja), la GPU supera ampliamente a la CPU gracias a su mayor ancho de banda de memoria (GDDR6 vs DDR4). El *overhead* de transferencia `cudaMemcpy` penaliza a la GPU para N pequeños, pero a 10 millones de elementos la ventaja es clara.

> 📸 **Evidencia:** 
![Ejercicio 8](ejercicio8_tiempo/img/ejercicio8_tiempo.png)
---

## Ejercicio 9 — Producto Punto de Vectores 

### ¿Qué hace el código?
Calcula el producto punto `∑ a[i]·b[i]` usando el mismo patrón de reducción del Ejercicio 7, pero con un paso previo de multiplicación. **Caso 1:** vectores de `1.0f`, resultado esperado `N = 4096`. **TAREA:** vectores aleatorios con verificación contra resultado de CPU usando tolerancia relativa `1e-4`.

### Comandos
```bash
nvcc ejercicio9_producto_punto.cu -o ejercicio9 -lm
./ejercicio9
```

### Salida esperada
```
=== Caso 1: vectores de 1.0f (resultado esperado = 4096) ===
Producto punto (GPU) = 4096.00
Esperado             = 4096
[OK]

=== TAREA: vectores aleatorios (verificacion GPU vs CPU) ===
Producto punto (CPU referencia) = 1023.847412
Producto punto (GPU)            = 1023.851318
Diferencia absoluta             = 0.003906  (tolerancia 0.102385)
[OK] Resultados coinciden!
```

### TAREA — Vectores aleatorios y tolerancia
El resultado de la CPU se calcula en `double` para maximizar precisión y servir como referencia. La GPU opera en `float` (32 bits), y la suma de 4096 multiplicaciones acumula error de redondeo. Por eso se usa tolerancia relativa `fabsf(ref) * 1e-4f` en lugar de comparar exactamente. Esta diferencia no es un error del programa sino una propiedad fundamental de la aritmética en punto flotante.

> 📸 **Evidencia:** 
![Ejercicio 9](ejercicio9_producto_punto/img/ejercicio9_producto_punto.png)
---

## Reflexión Final

| Concepto aprendido | Ejercicios |
|--------------------|-----------|
| Flujo CPU→GPU→CPU con `cudaMalloc` / `cudaMemcpy` / `cudaFree` | 1, 2 |
| Consulta de capacidades del hardware con `cudaGetDeviceProperties` | 3 |
| Diseño de kernels 1D con índice global `blockIdx.x * blockDim.x + threadIdx.x` | 4, 5 |
| Kernels 2D con `dim3` e indexación por filas y columnas | 6 |
| Reducción paralela y uso correcto de `shared memory` + `__syncthreads()` | 7, 9 |
| Medición de rendimiento con CUDA Events y cálculo de bandwidth | 8 |

---

*Taller elaborado para el curso de Programación Paralela y Computación Distribuida — Semestre 2026-I*  
*Prf. Juan Alejandro Carrillo Jaimes — Universidad de Pamplona, Colombia*
