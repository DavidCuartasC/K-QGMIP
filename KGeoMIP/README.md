# KGeoMIP — GeoMIP extendido a k‑particiones

Proyecto **K‑QGMIP** · Análisis y Diseño de Algoritmos · Universidad de Caldas · 2026‑1

## Descripción

Extensión de la estrategia geométrica **GeoMIP** del problema de la **Partición
de Mínima Información (MIP)** del caso de bi‑particiones (k = 2) al caso general
de **k‑particiones** (k ∈ {2, 3, 4, 5}). La clase principal es **`KGeoMIP`**.

> La extensión análoga de QNodes, **`KQNodes`**, está **PENDIENTE** (no implementada).

## Objetivo

Dado un sistema binario y su matriz de transición (TPM), hallar la **k‑partición
de mínima información** (k‑MIP): la división del sistema en `k` partes que
minimiza la pérdida de información causal δₖ (EMD‑Efecto).

## Estrategias

| Estrategia | Clase | Estado |
|---|---|---|
| GeoMIP (bipartición) | `GeometricSIA` | Original |
| **KGeoMIP (k‑particiones)** | `KGeoMIP` | ✅ Implementada y validada |
| QNodes (bipartición) | `QNodes` | Original |
| KQNodes (k‑particiones) | — | ⏳ PENDIENTE |

## Requisitos

- Python ≥ 3.9 · [`uv`](https://docs.astral.sh/uv/) · NumPy, pandas, openpyxl.

## Instalación

```powershell
# desde KGeoMIP/src/Method2_Dynamic_Programming_Reformulation
uv run python -c "print('entorno listo')"   # uv crea el .venv e instala deps
```

## Uso básico

```powershell
# Demo (k vía $env:KGEOMIP_K, por defecto 3)
uv run python -c "from src.main import iniciar_kgeomip; iniciar_kgeomip()"

# Llamada directa
uv run python -c "from src.main import ejecutar_kgeomip; ejecutar_kgeomip('100','111','111', k=2)"
```

## Estructura de carpetas

```
KGeoMIP/
├── data/samples/            # TPMs de prueba (N3A.csv, N4A.csv, …)
├── docs/                    # Documentación del proyecto
│   ├── manual_tecnico/      # Manual Técnico
│   ├── manual_usuario/      # Manual de Usuario
│   ├── diagramas/           # Diagramas UML (Mermaid)
│   ├── resultados/          # Benchmark, CSV y gráficas
│   ├── video/               # Guion del video tutorial
│   └── presentacion/        # Guion de la presentación
└── src/Method2_Dynamic_Programming_Reformulation/
    ├── src/controllers/strategies/kgeomip.py   # KGeoMIP
    ├── src/models/core/system.py               # k_partir
    ├── src/funcs/format.py                      # fmt_k_particion
    └── tests/test_kgeomip.py                    # 12 tests
```

## Pruebas

```powershell
uv run --with pytest pytest tests/ -v     # 12 passed
```

## Resultados

```powershell
uv run python ../../docs/resultados/benchmark.py                 # tabla CSV
uv run --with matplotlib python ../../docs/resultados/graficas.py # PNGs
```

Resumen: consistencia k=2 con GeoMIP al 100 % (N=3,4,5,6,8); pérdida monótona en
k; exacto y rápido para N ≤ 6. Ver [`docs/resultados/`](docs/resultados/).

## Autores

`[PENDIENTE: nombres del equipo]`

## Estado del proyecto

KGeoMIP funcional y probado. Documentación base completa. KQNodes y modo exacto
para sistemas grandes: pendientes.

## Limitaciones conocidas

- B&B sin garantía de óptimo global en sistemas grandes (modo heurístico al
  alcanzar el tope de nodos).
- Asume subsistema completo (|presente| = |futuro| = N).
- KQNodes no implementado.
