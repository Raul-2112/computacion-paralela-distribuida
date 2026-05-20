# lab_cuda_basic_intermediate | Valentina Cerpa & Raul Delgado

> **Asignatura:** Fundamentos de Programación Concurrente y Distribuida  
> **Docente:** Prf. Alejandro Jaimes  
> **Fecha:** 18/05/2026  
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
## Descripción

Este repositorio contiene la solución completa del taller de introducción a CUDA, organizado en 9 ejercicios progresivos que cubren desde transferencias de memoria básicas hasta kernels con shared memory y medición de rendimiento.

## Estructura del repositorio

```
taller-cuda-pareja-XX/
├── README.md                          # Este archivo
├── reporte/
│   └── taller_cuda.md                 # Reporte con evidencias
├── ejercicio1_hola_gpu/
│   ├── ejercicio1_hola_gpu.cu
│   └── README.md
├── ejercicio2_matriz/
│   ├── ejercicio2_matriz.cu
│   └── README.md
├── ejercicio3_device_info/
│   ├── ejercicio3_device_info.cu
│   └── README.md
├── ejercicio4_suma_vectores/
│   ├── ejercicio4_suma_vectores.cu
│   └── README.md
├── ejercicio5_cuadrado/
│   ├── ejercicio5_cuadrado.cu
│   └── README.md
├── ejercicio6_kernel2d/
│   ├── ejercicio6_kernel2d.cu
│   └── README.md
├── ejercicio7_reduccion/
│   ├── ejercicio7_reduccion.cu
│   └── README.md
├── ejercicio8_tiempo/
│   ├── ejercicio8_tiempo.cu
│   └── README.md
└── ejercicio9_producto_punto/
    ├── ejercicio9_producto_punto.cu
    └── README.md
```

## Requisitos

- NVIDIA GPU con soporte CUDA (Compute Capability ≥ 3.0)
- CUDA Toolkit ≥ 11.0
- Compilador: `nvcc`
- SO: Linux o Windows con drivers NVIDIA instalados

## Compilación rápida de todos los ejercicios

```bash
# Linux
for i in 1 2 3 4 5 6 7 8 9; do
  cd ejercicio${i}_*/
  nvcc *.cu -o ejercicio${i} && echo "OK: ejercicio${i}"
  cd ..
done
```

## Resumen de ejercicios

| # | Nombre | Categoría | Concepto clave |
|---|--------|-----------|----------------|
| 1 | Hola GPU | Transferencia | `cudaMalloc`, `cudaMemcpy` |
| 2 | Copia Matriz 2D | Transferencia | Arreglo 2D aplanado |
| 3 | Device Info | Transferencia | `cudaGetDeviceProperties` |
| 4 | Suma de Vectores | Kernel básico | Índice global, grilla 1D |
| 5 | Cuadrado In-place | Kernel básico | Escritura sobre mismo buffer |
| 6 | Kernel 2D | Kernel básico | `dim3`, índice 2D |
| 7 | Reducción | Intermedio | Shared memory, `__syncthreads` |
| 8 | Tiempo Escalar | Intermedio | CUDA Events, bandwidth |
| 9 | Producto Punto | Intermedio | Reducción + multiplicación |
