"""Almacén columnar de TPM para análisis de redes grandes (streaming).

La TPM se guarda como un arreglo ``(N, 2^N)`` (una **fila contigua por nodo** =
la columna de ese nodo en la TPM original ``(2^N, N)``), en ``int8`` si la red es
binaria (determinista 0/1) o ``float32`` si es estocástica. Se accede por memmap,
de modo que solo se carga en RAM la columna que se está usando.

Esto evita los dos muros de memoria de N grande:
  - No se carga la TPM completa en float64 (6.7 GB para N=25).
  - No se construyen los N n-cubos a la vez (otros 6.7 GB).
El pico de RAM es ~una columna (2^25 int8 = 33.5 MB).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


class ColumnStore:
    """Acceso por memmap a una TPM almacenada como (N, 2^N)."""

    def __init__(self, path: str | Path):
        # mmap_mode='r': las filas se leen de disco bajo demanda.
        self._arr = np.load(path, mmap_mode="r")
        if self._arr.ndim != 2:
            raise ValueError(f"Se esperaba (N, 2^N); recibido {self._arr.shape}")
        self.n_nodes: int = int(self._arr.shape[0])
        self.n_estados: int = int(self._arr.shape[1])
        self.dim: int = int(round(math.log2(self.n_estados)))
        if (1 << self.dim) != self.n_estados:
            raise ValueError(f"n_estados={self.n_estados} no es potencia de 2")
        self.shape_ncubo: tuple[int, ...] = (2,) * self.dim

    def columna(self, i: int) -> np.ndarray:
        """Columna plana del nodo i (vista memmap de longitud 2^N)."""
        return self._arr[i]

    def columna_ncubo(self, i: int) -> np.ndarray:
        """Columna del nodo i con forma (2,)*N (vista, sin copia)."""
        return self._arr[i].reshape(self.shape_ncubo)
