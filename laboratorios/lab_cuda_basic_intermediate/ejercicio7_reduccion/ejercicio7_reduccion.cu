// Archivo: ejercicio7_reduccion.cu
// Objetivo: Suma de arreglo con reducción paralela y shared memory
#include <stdio.h>
#include <cuda_runtime.h>

#define N       1024
#define THREADS  256

// Kernel de reducción usando shared memory
__global__ void reduccionSuma(int *d_entrada, int *d_salida, int n) {
    extern __shared__ int s_datos[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Cargar datos de memoria global a shared memory
    s_datos[tid] = (idx < n) ? d_entrada[idx] : 0;
    __syncthreads();

    // Reducción: stride empieza en la mitad del bloque
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_datos[tid] += s_datos[tid + stride];
        }
        __syncthreads();
    }

    // El hilo 0 escribe el resultado parcial del bloque
    if (tid == 0) {
        d_salida[blockIdx.x] = s_datos[0];
    }
}

int main() {
    int h_datos[N];
    int suma_cpu = 0;
    for (int i = 0; i < N; i++) { h_datos[i] = 1; suma_cpu += h_datos[i]; }
    printf("Suma esperada (CPU): %d\n", suma_cpu);

    int numBloques = (N + THREADS - 1) / THREADS;
    int h_parciales[numBloques];

    int *d_datos, *d_parciales;
    cudaMalloc((void**)&d_datos,    N         * sizeof(int));
    cudaMalloc((void**)&d_parciales, numBloques * sizeof(int));
    cudaMemcpy(d_datos, h_datos, N * sizeof(int), cudaMemcpyHostToDevice);

    // Tercer parámetro: tamaño de shared memory en bytes por bloque
    int sharedBytes = THREADS * sizeof(int);
    reduccionSuma<<<numBloques, THREADS, sharedBytes>>>(d_datos, d_parciales, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_parciales, d_parciales, numBloques * sizeof(int), cudaMemcpyDeviceToHost);

    // Suma de resultados parciales en CPU
    int suma_gpu = 0;
    for (int i = 0; i < numBloques; i++) suma_gpu += h_parciales[i];
    printf("Suma calculada (GPU): %d\n", suma_gpu);
    printf("%s\n", (suma_cpu == suma_gpu) ? "[OK] Resultados identicos!" : "[ERROR] No coinciden.");

    printf("\nDetalle de %d sumas parciales (una por bloque):\n", numBloques);
    for (int i = 0; i < numBloques; i++)
        printf("  Bloque %d -> parcial = %d\n", i, h_parciales[i]);

    cudaFree(d_datos); cudaFree(d_parciales);
    return 0;
}

// Compilar: nvcc ejercicio7_reduccion.cu -o ejercicio7
// Nota: __syncthreads() es CRUCIAL. Sin él, los hilos leerían datos incorrectos.
