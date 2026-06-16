"""Tests de KGeoMIP (extensión de GeoMIP a k-particiones).

Qué valida cada test:
  - test_validacion_entradas: k<2 y k>N lanzan ValueError (robustez de inputs).
  - test_consistencia_k2_N3 / _N4: para k=2, KGeoMIP reproduce la pérdida de la
    bipartición de GeoMIP original (requisito del proyecto: consistencia k=2).
  - test_perdida_no_negativa: δₖ >= 0 siempre.
  - test_monotonia_k: más partes ⇒ pérdida mayor o igual (separar más nunca
    reduce la pérdida de información).
  - test_estructura_kparticion: la k-partición cubre todos los vértices, sin
    solapamiento y con k partes no vacías (definición formal de k-partición).

Ejecución (desde la raíz de Method2_Dynamic_Programming_Reformulation):
    uv run --with pytest pytest tests/ -v
"""
import numpy as np
import pytest

from src.constants.base import ACTUAL, EFECTO
from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.kgeomip import KGeoMIP
from src.main import resolver_tpm_path

TOL = 1e-6


def _tpm(estado: str) -> np.ndarray:
    return np.genfromtxt(resolver_tpm_path(estado), delimiter=",")


def _kgeomip(estado: str, alcance: str, mecanismo: str, k: int):
    tpm = _tpm(estado)
    est = KGeoMIP(Manager(estado_inicial=estado))
    sol = est.aplicar_estrategia_k("1" * len(estado), alcance, mecanismo, tpm, k)
    return est, sol


def _geomip_biparticion(estado: str, alcance: str, mecanismo: str) -> float:
    tpm = _tpm(estado)
    est = GeometricSIA(Manager(estado_inicial=estado))
    return est.aplicar_estrategia("1" * len(estado), alcance, mecanismo, tpm).perdida


# --------------------------------------------------------------------------- #
# Validación de entradas
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("k", [0, 1])
def test_validacion_k_menor_2(k):
    with pytest.raises(ValueError):
        _kgeomip("100", "111", "111", k)


def test_validacion_k_mayor_pf():
    # k > p+f (vértices del subsistema bipartito) debe lanzar ValueError.
    # "100" completo: f=3, p=3 → p+f=6, así que k=7 es inválido.
    with pytest.raises(ValueError):
        _kgeomip("100", "111", "111", 7)


# --------------------------------------------------------------------------- #
# Consistencia con GeoMIP original para k=2
# --------------------------------------------------------------------------- #

def test_consistencia_k2_N3():
    _, sol = _kgeomip("100", "111", "111", 2)
    geo = _geomip_biparticion("100", "111", "111")
    assert sol.perdida == pytest.approx(geo, abs=TOL)


def test_consistencia_k2_N4():
    _, sol = _kgeomip("1000", "1111", "1111", 2)
    geo = _geomip_biparticion("1000", "1111", "1111")
    assert sol.perdida == pytest.approx(geo, abs=TOL)


# --------------------------------------------------------------------------- #
# Propiedades de la pérdida δₖ
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("k", [2, 3])
def test_perdida_no_negativa(k):
    _, sol = _kgeomip("100", "111", "111", k)
    assert sol.perdida >= -TOL


def test_monotonia_k():
    p2 = _kgeomip("1000", "1111", "1111", 2)[1].perdida
    p3 = _kgeomip("1000", "1111", "1111", 3)[1].perdida
    p4 = _kgeomip("1000", "1111", "1111", 4)[1].perdida
    assert p2 <= p3 + TOL
    assert p3 <= p4 + TOL


# --------------------------------------------------------------------------- #
# Estructura de la k-partición
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("estado,k", [("100", 2), ("100", 3), ("1000", 3), ("1000", 4)])
def test_estructura_kparticion(estado, k):
    est, _ = _kgeomip(estado, "1" * len(estado), "1" * len(estado), k)
    A = est._construir_afinidad()
    kpart = est._asignacion_a_kparticion(est._greedy(k, A))

    esperado = {(EFECTO, int(i)) for i in est._futuros} | {
        (ACTUAL, int(d)) for d in est._presentes
    }
    cubiertos = set().union(*kpart)

    assert cubiertos == esperado                              # cobertura total
    assert sum(len(g) for g in kpart) == len(esperado)        # sin solapamiento
    assert all(len(g) > 0 for g in kpart)                     # partes no vacías
    assert len(kpart) == k                                    # exactamente k partes


# --------------------------------------------------------------------------- #
# Modos de ejecución, telemetría y calidad heurística (Nivel 1)
# --------------------------------------------------------------------------- #

def _kgeomip_modo(estado: str, k: int, mode: str, **kw):
    tpm = _tpm(estado)
    n = len(estado)
    est = KGeoMIP(Manager(estado_inicial=estado))
    return est.aplicar_estrategia_k("1" * n, "1" * n, "1" * n, tpm, k, mode=mode, **kw)


def test_mode_invalido():
    with pytest.raises(ValueError):
        _kgeomip_modo("100", 2, "xyz")


def test_telemetria_presente():
    sol = _kgeomip_modo("100000", 3, "auto")
    assert sol.status in {"optimal", "capped", "timeout", "heuristic", "infeasible"}
    assert sol.modo_ejecucion in {"auto", "exact", "bnb", "heuristic", "infeasible"}
    assert isinstance(sol.nodos_explorados, int)


def test_exact_es_optimal():
    sol = _kgeomip_modo("100000", 3, "exact")
    assert sol.status == "optimal"


def test_status_capped_con_tope_minimo():
    # Con un tope de 1 nodo el B&B se trunca: debe reportar 'capped' y aún así
    # devolver una k-partición válida (la semilla greedy).
    sol = _kgeomip_modo("100000", 3, "bnb", max_nodos=1)
    assert sol.status == "capped"
    assert sol.perdida >= -TOL


@pytest.mark.parametrize(
    "estado,k", [("100", 3), ("10000", 3), ("100000", 3), ("100000", 4)]
)
def test_heuristico_no_mejora_optimo(estado, k):
    # La heurística nunca debe reportar MENOS pérdida que el óptimo exacto.
    exacto = _kgeomip_modo(estado, k, "exact").perdida
    heur = _kgeomip_modo(estado, k, "heuristic").perdida
    assert heur >= exacto - TOL


@pytest.mark.parametrize("mode", ["exact", "bnb", "heuristic", "auto"])
def test_modos_devuelven_resultado_valido(mode):
    sol = _kgeomip_modo("100000", 3, mode)
    assert sol.perdida >= -TOL
    assert isinstance(sol.particion, str) and sol.particion.strip()


# --------------------------------------------------------------------------- #
# Subsistemas incompletos (|alcance| != |mecanismo|)
# --------------------------------------------------------------------------- #

def test_subsistema_incompleto_funciona():
    # alcance lleno (f=5), mecanismo parcial (p=3): antes lanzaba NotImplementedError.
    tpm = _tpm("10000")
    est = KGeoMIP(Manager(estado_inicial="10000"))
    sol = est.aplicar_estrategia_k("11111", "11111", "11100", tpm, 2)
    assert sol.perdida >= -TOL
    assert isinstance(sol.particion, str) and sol.particion.strip()


def test_exacto_incompleto_es_optimo_global():
    # KGeoMIP exacto k=2 sobre un subsistema incompleto debe igualar el mínimo
    # por enumeración exhaustiva de todas las biparticiones de los p+f vértices.
    from itertools import product

    tpm = _tpm("10000")
    est = KGeoMIP(Manager(estado_inicial="10000"))
    sol = est.aplicar_estrategia_k("11111", "11111", "11100", tpm, 2, mode="exact")

    mejor = float("inf")
    for bits in product((0, 1), repeat=est._M):
        if len(set(bits)) != 2:  # ambas partes no vacías
            continue
        asig = {lvl: bits[lvl] for lvl in range(est._M)}
        emd, _ = est._evaluar_kparticion(est._asignacion_a_kparticion(asig))
        mejor = min(mejor, emd)

    assert sol.perdida == pytest.approx(mejor, abs=TOL)
