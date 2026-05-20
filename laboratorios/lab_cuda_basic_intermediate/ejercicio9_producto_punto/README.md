# Ejercicio 9 — Producto Punto de Vectores

## Descripción
Calcula el producto punto `∑ a[i]·b[i]` combinando multiplicación elemento a elemento y reducción paralela con shared memory. Incluye dos casos de prueba: vectores de unos y vectores aleatorios con verificación contra resultado de referencia en CPU.

## Archivo
- `ejercicio9_producto_punto.cu`

## Conceptos aplicados
- Kernel que fusiona multiplicación y reducción en un solo paso
- Shared memory + reducción en árbol (patrón del Ejercicio 7)
- Suma de parciales por bloque en CPU
- Tolerancia relativa para comparar resultados en `float` vs `double`

## TAREA resuelta
Prueba con vectores generados con `rand()`. El valor de referencia se calcula en `double` en la CPU para maximizar precisión. La verificación usa tolerancia relativa `1e-4` porque las operaciones en `float` acumulan error de redondeo al sumar miles de términos.

```c
float tolerancia = fabsf((float)ref) * 1e-4f;
float diff = fabsf(resultado_gpu - (float)ref);
// OK si diff <= tolerancia
```

## Compilar y ejecutar

```bash
nvcc ejercicio9_producto_punto.cu -o ejercicio9 -lm
./ejercicio9
```

## Salida esperada

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

## Evidencia

> *(adjuntar pantallazo de compilación y ejecución)*
