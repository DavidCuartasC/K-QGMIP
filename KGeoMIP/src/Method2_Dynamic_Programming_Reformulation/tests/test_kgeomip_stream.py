"""Tests del modo streaming (KGeoMIPStream) para redes grandes.

Qué validan:
  - test_stream_coincide_exacto: en casos pequeños donde la heurística alcanza el
    óptimo, el δ del modo stream coincide con el exacto (valida que la
    marginalización por columnas en streaming es matemáticamente correcta).
  - test_stream_no_mejora_optimo: el stream nunca reporta menos que el óptimo.
  - test_stream_status_particion: status='heuristic', modo='stream', partición válida.
  - test_stream_validaciones: subsistema incompleto → NotImplementedError; k inválido.

El column store se construye desde el CSV existente en un tmp_path (tests
autocontenidos; no dependen de .npy pre-generados).

Ejecución: uv run --with pytest pytest tests/test_kgeomip_stream.py -v
"""
import numpy as np
import pytest

from src.controllers.manager import Manager
from src.controllers.strategies.kgeomip import KGeoMIP
from src.controllers.strategies.kgeomip_stream import KGeoMIPStream
from src.funcs.tpm_store import ColumnStore
from src.main import resolver_tpm_path

TOL = 1e-6


def _store(estado: str, tmp_path) -> ColumnStore:
    tpm = np.genfromtxt(resolver_tpm_path(estado), delimiter=",")
    cols = np.ascontiguousarray(tpm.T.astype(np.int8))  # (N, 2^N)
    p = tmp_path / f"N{len(estado)}_cols.npy"
    np.save(p, cols)
    return ColumnStore(p)


def _exact(estado: str, k: int) -> float:
    n = len(estado)
    tpm = np.genfromtxt(resolver_tpm_path(estado), delimiter=",")
    sol = KGeoMIP(Manager(estado_inicial=estado)).aplicar_estrategia_k(
        "1" * n, "1" * n, "1" * n, tpm, k, mode="exact"
    )
    return float(sol.perdida)


def _stream(estado: str, k: int, store: ColumnStore, n_starts: int = 8):
    n = len(estado)
    return KGeoMIPStream(Manager(estado_inicial=estado)).aplicar_estrategia_stream(
        store, estado, "1" * n, "1" * n, k, n_starts=n_starts
    )


# Casos donde la heurística (determinista, semillas fijas) alcanza el óptimo.
@pytest.mark.parametrize(
    "estado,k", [("100000", 2), ("100000", 3), ("1000000000", 2), ("1000000000", 3)]
)
def test_stream_coincide_exacto(estado, k, tmp_path):
    store = _store(estado, tmp_path)
    st = float(_stream(estado, k, store).perdida)
    ex = _exact(estado, k)
    assert st >= ex - TOL               # nunca mejor que el óptimo
    assert st == pytest.approx(ex, abs=TOL)  # y aquí lo alcanza


def test_stream_status_particion(tmp_path):
    store = _store("100000", tmp_path)
    sol = _stream("100000", 3, store)
    assert sol.status == "heuristic"
    assert sol.modo_ejecucion == "stream"
    assert sol.perdida >= -TOL
    assert isinstance(sol.particion, str) and sol.particion.strip()


def test_stream_subsistema_incompleto_funciona(tmp_path):
    # f=6 futuros, p=5 presentes (incompleto): debe funcionar y dar un resultado válido.
    store = _store("100000", tmp_path)
    est = KGeoMIPStream(Manager(estado_inicial="100000"))
    sol = est.aplicar_estrategia_stream(store, "100000", "111111", "111110", 2)
    assert sol.status == "heuristic"
    assert sol.perdida >= -TOL
    assert isinstance(sol.particion, str) and sol.particion.strip()


@pytest.mark.parametrize("k", [1, 13])
def test_stream_k_invalido(k, tmp_path):
    # k<2 o k>p+f (aquí p+f=12 para subsistema completo de m=6) → ValueError.
    store = _store("100000", tmp_path)
    est = KGeoMIPStream(Manager(estado_inicial="100000"))
    with pytest.raises(ValueError):
        est.aplicar_estrategia_stream(store, "100000", "111111", "111111", k)
