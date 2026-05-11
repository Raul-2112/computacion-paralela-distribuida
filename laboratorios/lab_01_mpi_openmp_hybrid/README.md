# LAB-01-MPI-OPENMP-HYBRID | Valentina Cerpa & Raul Delgado

> **Asignatura:** Fundamentos de Programación Concurrente y Distribuida  
> **Docente:** Prf. Alejandro Jaimes  
> **Fecha:** 11/05/2026  
> **Repositorio:** [LINK_REPO](https://github.com/Raul-2112/computacion-paralela-distribuida)

---

## Equipo

| | Colaborador | GitHub |
|---|---|---|
| 👤 | Valentina Cerpa | [@Valen187](https://github.com/Valen187) |
| 👤 | Raul Delgado | [@Raul-2112](https://github.com/Raul-2112) |

**Repositorio:** [computacion-paralela-distribuida](https://github.com/Raul-2112/computacion-paralela-distribuida)  
**Rama principal:** `main`

---

# Ejercicio 1: Hola Mundo MPI

## Código fuente
`mpi_01_hola.c`

## Compilación
```bash
mpicc mpi_01_hola.c -o mpi_01_hola.exe
```

## Ejecución
```bash
mpiexec -n 4 .\mpi_01_hola.exe
mpiexec -n 2 .\mpi_01_hola.exe
```

## Evidencia
![Ejercicio 1](img/ejecucion_ej1.png)
---

# Ejercicio 2: Programación Híbrida MPI + OpenMP

## Código fuente
`mpi_02_Hibrido.c`

## Compilación
```bash
mpicc mpi_02_Hibrido.c -o mpi_02_Hibrido.exe -fopenmp
```

## Ejecución
```bash
mpiexec -n 4 mpi_02_Hibrido.exe
```

## Evidencia
![Ejercicio 2](img/ejecucion_ej2.png)

---

# Ejercicio 3: Suma hibrida de vector 

## Código fuente
`mpi_03_suma_hibrida.c`

## Compilación
```bash
mpicc -fopenmp mpi_03_suma_hibrida.c -o mpi_03.exe
```

## Ejecución
```bash
mpiexec -n 4 .\mpi_03.exe
```

## Evidencia
![Ejercicio 3](img/ejecucion_ej3.png)

---

# Ejercicio 4: Medición de Speedup

## Código fuente
`mpi_04_Speedup.c`

## Compilación
```bash
.\mpicc mpi_04_Speedup.c -o mpi_04_speedup.exe -fopenmp
```

## Ejecución
```bash
mpiexec -n 4 mpi_04_speedup.exe
```

## Evidencia
![Ejercicio 4](img/ejecucion_ej4.png)

---

## Tecnologías utilizadas
- C
- Microsoft MPI
- OpenMP
- MinGW-w64
- Visual Studio Code
- GitHub