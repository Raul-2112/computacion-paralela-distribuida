#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_threads = 0;

    // TODO 1: Región paralela OpenMP
    #pragma omp parallel num_threads(4)
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        // Guardamos número de hilos (solo una vez)
        #pragma omp single
        {
            num_threads = total_threads;
        }

        printf("Proceso MPI %d | Hilo OpenMP %d de %d\n",
               rank, thread_id, total_threads);
    }

    // TODO 2: Solo el proceso 0 imprime el total
    if (rank == 0) {
        printf("\nTotal unidades: %d procesos MPI x %d hilos OMP = %d\n",
               size, num_threads, size * num_threads);
    }

    MPI_Finalize();
    return 0;
}