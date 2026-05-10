#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = N / size;
    long long *arr = NULL;

    // ── Inicio de medición del tiempo paralelo (rank 0) ──────────
    double t_inicio = MPI_Wtime();

    // TODO 1: Solo rank==0 crea y llena el arreglo completo
    if (rank == 0) {
        arr = (long long*) malloc(N * sizeof(long long));
        for (int i = 0; i < N; i++) arr[i] = i;
    }

    // Cada proceso recibe su porción del arreglo
    long long *local = (long long*) malloc(chunk * sizeof(long long));

    // TODO 2: MPI_Scatter
    MPI_Scatter(arr, chunk, MPI_LONG_LONG, local, chunk, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // TODO 3: Suma local con OpenMP
    long long suma_local = 0;
    #pragma omp parallel for reduction(+:suma_local)
    for (int i = 0; i < chunk; i++)
        suma_local += local[i];

    // TODO 4: MPI_Reduce
    long long suma_total = 0;
    MPI_Reduce(&suma_local, &suma_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // ── Fin de medición del tiempo paralelo ───────────────────────
    double t_fin = MPI_Wtime();

    if (rank == 0) {
        printf("Suma total = %lld\n", suma_total);
        printf("Esperado  = %lld\n", (long long)N*(N-1)/2);
        printf("Tiempo paralelo: %.4f segundos\n", t_fin - t_inicio);
    }

    // ── Versión secuencial (solo rank==0) ────────────────────────
    if (rank == 0) {
        long long suma_seq = 0;
        double ts = MPI_Wtime();

        // TODO: for secuencial — mismo cálculo sin MPI ni OpenMP
        for (int i = 0; i < N; i++)
            suma_seq += i;

        double te = MPI_Wtime();

        printf("Suma secuencial= %lld\n", suma_seq);
        printf("Tiempo secuencial: %.4f segundos\n", te - ts);
        printf("Speedup: %.2fx\n", (te - ts) / (t_fin - t_inicio));

        free(arr);
    }

    free(local);
    MPI_Finalize();
    return 0;
}