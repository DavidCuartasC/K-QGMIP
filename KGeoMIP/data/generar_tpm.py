"""Genera/convierte TPMs al formato columnar binario (N, 2^N) para N grande.

Dos modos:
  - generar:  red sintética aleatoria (0/1) con semilla fija, escrita fila a fila
              (memoria baja: una columna 2^N a la vez). Para N=20/22/25 que no
              existen como dataset.
  - desde-csv: convierte un N{n}{page}.csv existente (2^N, N) al formato columnar
              (para validar el modo streaming contra el exacto, o reutilizar datos).

Salida: data/samples/N{n}{page}_cols.npy  (int8 si binaria; float32 si estocástica).

Uso (desde KGeoMIP/data):
    uv run python generar_tpm.py generar --n 25 --seed 0
    uv run python generar_tpm.py desde-csv --csv samples/N10A.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

KGEOMIP_ROOT = Path(__file__).resolve().parents[1]  # .../KGeoMIP (este archivo: KGeoMIP/data/)
SAMPLES = KGEOMIP_ROOT / "data" / "samples"


def _salida(n: int, page: str) -> Path:
    return SAMPLES / f"N{n}{page}_cols.npy"


def generar(n: int, seed: int, page: str = "A", discreto: bool = True) -> Path:
    """Genera una red sintética (N, 2^N) fila a fila (baja memoria)."""
    n_estados = 1 << n
    dtype = np.int8 if discreto else np.float32
    salida = _salida(n, page)
    salida.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(salida, mode="w+", dtype=dtype, shape=(n, n_estados))
    rng = np.random.default_rng(seed)
    for i in range(n):
        if discreto:
            arr[i] = rng.integers(0, 2, size=n_estados, dtype=np.int8)
        else:
            arr[i] = rng.random(n_estados, dtype=np.float32)
        print(f"  nodo {i + 1}/{n} generado")
    arr.flush()
    print(f"Guardado: {salida}  ({dtype.__name__}, {arr.nbytes / 1e6:.1f} MB)")
    return salida


def desde_csv(csv_path: Path) -> Path:
    """Convierte N{n}{page}.csv (2^N, N) a formato columnar (N, 2^N)."""
    tpm = np.genfromtxt(csv_path, delimiter=",")  # (2^N, N) float64
    n = int(tpm.shape[1])
    binaria = bool(np.all((tpm == 0) | (tpm == 1)))
    dtype = np.int8 if binaria else np.float32
    cols = np.ascontiguousarray(tpm.T.astype(dtype))  # (N, 2^N), filas contiguas
    nombre = csv_path.stem  # p. ej. "N10A"
    salida = SAMPLES / f"{nombre}_cols.npy"
    np.save(salida, cols)
    print(f"Guardado: {salida}  (N={n}, {dtype.__name__}, binaria={binaria}, "
          f"{cols.nbytes / 1e6:.1f} MB)")
    return salida


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera/convierte TPMs columnar.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generar", help="red sintética aleatoria.")
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--page", default="A")
    g.add_argument("--estocastica", action="store_true", help="float32 en vez de 0/1.")

    c = sub.add_parser("desde-csv", help="convierte un CSV existente.")
    c.add_argument("--csv", required=True)

    args = ap.parse_args()
    if args.cmd == "generar":
        generar(args.n, args.seed, args.page, discreto=not args.estocastica)
    else:
        desde_csv(Path(args.csv))


if __name__ == "__main__":
    main()
