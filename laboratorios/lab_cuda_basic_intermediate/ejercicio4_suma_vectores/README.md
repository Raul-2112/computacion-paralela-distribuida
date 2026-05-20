# Ejercicio 4 — Suma de Vectores Paralela

## Descripción
El "Hola Mundo" de CUDA. Dos vectores de 1 000 000 de elementos se suman en paralelo en la GPU, donde cada hilo calcula exactamente un par de elementos.

## Archivo
- `ejercicio4_suma_vectores.cu`

## Conceptos aplicados
- Diseño de kernel `__global__` con índice global 1D
- Cálculo de dimensiones: `numBloques = (N + THREADS - 1) / THREADS`
- Guard de límites: `if (idx < n)` para hilos sobrantes
- `cudaDeviceSynchronize` — esperar a que el kernel termine

## Fórmula del índice global

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

## Compilar y ejecutar

```bash
nvcc ejercicio4_suma_vectores.cu -o ejercicio4
./ejercicio4
```

## Salida esperada

```
Lanzando 3907 bloques x 256 hilos = 1000192 hilos totales
h_C[0]   = 3.0 (esperado: 3.0)
h_C[N-1] = 3.0 (esperado: 3.0)

[OK] Suma de vectores completada.
```

## Evidencia

> *(adjuntar pantallazo de compilación y ejecución)*
