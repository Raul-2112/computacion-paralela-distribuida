// Archivo: ejercicio9_producto_punto.cu
// Objetivo: Producto punto combinando multiplicación en GPU y reducción
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N       4096
#define THREADS  256

// Kernel: multiplicación elemento a elemento + reducción por bloque
__global__ void productoPunto(float *d_A, float *d_B, float *d_parciales, int n) {
    extern __shared__ float s_datos[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    s_datos[tid] = (idx < n) ? d_A[idx] * d_B[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            s_datos[tid] += s_datos[tid + stride];
        __syncthreads();
    }

    if (tid == 0) d_parciales[blockIdx.x] = s_datos[0];
}

int main() {
    // --- Prueba 1: vectores de 1.0f (resultado debe ser N) ---
    printf("=== Prueba 1: vectores de 1.0f ===\n");
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    int numBloques = (N + THREADS - 1) / THREADS;
    float *h_parciales = (float*)malloc(numBloques * sizeof(float));

    float *d_A, *d_B, *d_parciales;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_parciales, numBloques * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int sharedBytes = THREADS * sizeof(float);
    productoPunto<<<numBloques, THREADS, sharedBytes>>>(d_A, d_B, d_parciales, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_parciales, d_parciales, numBloques * sizeof(float), cudaMemcpyDeviceToHost);

    float resultado_gpu = 0.0f;
    for (int i = 0; i < numBloques; i++) resultado_gpu += h_parciales[i];

    printf("Producto punto GPU = %.2f\n", resultado_gpu);
    printf("Resultado esperado = %d\n", N);
    printf("%s\n", (resultado_gpu == (float)N) ? "[OK]" : "[ERROR]");

    // --- Prueba 2 (TAREA): vectores aleatorios, verificar contra CPU ---
    printf("\n=== Prueba 2 (TAREA): vectores aleatorios ===\n");
    srand(42);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;  // valor entre 0.0 y 1.0
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Calcular producto punto en CPU como referencia
    double referencia_cpu = 0.0;
    for (int i = 0; i < N; i++) referencia_cpu += (double)h_A[i] * h_B[i];

    // Enviar datos actualizados a GPU
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    productoPunto<<<numBloques, THREADS, sharedBytes>>>(d_A, d_B, d_parciales, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_parciales, d_parciales, numBloques * sizeof(float), cudaMemcpyDeviceToHost);

    float resultado_gpu2 = 0.0f;
    for (int i = 0; i < numBloques; i++) resultado_gpu2 += h_parciales[i];

    printf("Producto punto GPU  = %.6f\n", resultado_gpu2);
    printf("Referencia CPU      = %.6f\n", (float)referencia_cpu);

    float error_relativo = fabsf(resultado_gpu2 - (float)referencia_cpu)
                           / fabsf((float)referencia_cpu);
    printf("Error relativo      = %.2e\n", error_relativo);
    if (error_relativo < 1e-3f)
        printf("[OK] Error dentro de tolerancia (< 0.1%%).\n");
    else
        printf("[AVISO] Error mayor a tolerancia (diferencias de precision float).\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_parciales);
    free(h_A); free(h_B); free(h_parciales);
    return 0;
}

// Compilar: nvcc ejercicio9_producto_punto.cu -o ejercicio9 -lm
