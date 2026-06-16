"""KGeoMIPStream — KGeoMIP heurístico por *streaming* de columnas, para N grande.

Permite analizar redes donde construir el `System` completo (N n-cubos de 2^N) o la
tabla de costos sobre 2^N estados es inviable en memoria (N≈17..25). En vez de
materializar todo:

  - Lee la TPM desde un `ColumnStore` (memmap (N, 2^N)); solo una columna en RAM.
  - Evalúa cada k-partición **futuro a futuro** (δₖ es separable), reutilizando la
    marginalización de `NCube` sobre la columna en streaming, con caché.
  - Afinidad **aproximada** (media por columna) en una pasada → solo siembra el greedy.
  - Búsqueda **heurística** (greedy Welsh-Powell + búsqueda local move/swap +
    multi-start). NO usa B&B ni la tabla de costos. `status=heuristic` siempre.

Hereda de `KGeoMIP` para reutilizar `_greedy`, `_local_search`, `_grupos_disponibles`,
`_factible`, `_mecanismo`, `_asignacion_a_kparticion`. Sobrescribe la fuente de datos
(`_contrib_futuro`), la evaluación (`_evaluar_asignacion`) y la afinidad.
"""
from __future__ import annotations

import time

import numpy as np

from src.constants.models import KGEOMIP_LABEL
from src.controllers.strategies.kgeomip import KGeoMIP
from src.funcs.base import seleccionar_subestado
from src.funcs.format import fmt_k_particion
from src.funcs.tpm_store import ColumnStore
from src.models.core.ncube import NCube
from src.models.core.solution import Solution


class KGeoMIPStream(KGeoMIP):
    """Variante heurística por streaming de columnas (no construye el System)."""

    def aplicar_estrategia_stream(
        self,
        store: ColumnStore,
        estado: str,
        alcance: str,
        mecanismo: str,
        k: int,
        *,
        timeout: float | None = None,
        n_starts: int = 8,
        max_iter: int = 2000,
    ) -> Solution:
        """
        Halla una buena k-partición del subsistema (heurística), sin materializar
        el System completo.

        Args:
            store: ColumnStore con la TPM (N, 2^N).
            estado: estado inicial binario del sistema completo (longitud N).
            alcance: bits del alcance/futuro (1 = se conserva ese futuro).
            mecanismo: bits del mecanismo/presente (1 = se conserva ese presente).
            k: número de partes (2 <= k <= m, m = tamaño del subsistema).
            timeout: segundos máximos de búsqueda (None = sin límite).
            n_starts: reinicios de la búsqueda local (multi-start).
            max_iter: iteraciones máximas por búsqueda local.

        Returns:
            Solution con status='heuristic' (sin garantía de óptimo).
        """
        self._t0 = time.time()
        self._deadline = (self._t0 + timeout) if timeout else None
        N = store.n_nodes
        if len(estado) != N or len(alcance) != N or len(mecanismo) != N:
            raise ValueError(f"estado/alcance/mecanismo deben tener longitud N={N}")

        futuros = [i for i, b in enumerate(alcance) if b == "1"]   # f futuros
        presentes = [i for i, b in enumerate(mecanismo) if b == "1"]  # p presentes
        f, p = len(futuros), len(presentes)
        if k < 2:
            raise ValueError(f"k debe ser >= 2, recibido {k}")
        if k > f + p:
            raise ValueError(f"k={k} supera el nº de vértices del subsistema (p+f={f + p})")

        # --- Estado para el streaming (admite p != f) ---
        self._store = store
        self._estado = np.array([int(b) for b in estado], dtype=np.int8)
        self._dims_all = np.arange(N, dtype=np.int8)
        self._n_fut = f
        self._n_pre = p
        self._M = p + f
        self._futuros = futuros      # índices reales de nodo futuro
        self._presentes = presentes  # índices reales de dimensión presente
        self._orden = [("P", pos) for pos in range(p)] + [("F", pos) for pos in range(f)]
        self._cache_contrib = {}

        # dist_sub[pos] = marginal OFF del futuro pos conservando TODOS los
        # presentes del subsistema (mecanismo completo). Una pasada por columna.
        pres_arr = np.array(presentes, dtype=np.int8)
        self.sia_dists_marginales = np.array(
            [self._marginal_de(pos, pres_arr) for pos in range(f)],
            dtype=np.float64,
        )
        # Afinidad aproximada: media de actividad por columna (barata).
        self._sens = np.array(
            [float(self._store.columna(self._futuros[pos]).mean()) for pos in range(f)]
        )

        # --- Búsqueda heurística ---
        A = self._construir_afinidad()
        mejor = self._heuristico_stream(k, A, n_starts, max_iter, timeout)

        kpart = self._asignacion_a_kparticion(mejor)
        emd = self._evaluar_asignacion(mejor)
        dist = self._dist_particion(mejor)

        solucion = Solution(
            estrategia=f"{KGEOMIP_LABEL}-K{k} [heuristic]",
            perdida=emd,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            tiempo_total=time.time() - self._t0,
            particion=fmt_k_particion(kpart),
        )
        solucion.modo_ejecucion = "stream"
        solucion.status = "heuristic"
        solucion.nodos_explorados = 0
        return solucion

    # ------------------------------------------------------------------ #
    # Streaming de columnas                                                #
    # ------------------------------------------------------------------ #

    def _marginal_de(self, pos_fut: int, keep_dims: np.ndarray) -> float:
        """Marginal OFF (1 - P) del futuro pos_fut conservando solo `keep_dims`,
        seleccionado en el estado inicial. Lee una sola columna del store."""
        nodo = self._futuros[pos_fut]
        cube = NCube(
            indice=nodo,
            dims=self._dims_all,
            data=self._store.columna_ncubo(nodo),
        )
        cube_red = cube.marginalizar(np.setdiff1d(self._dims_all, keep_dims))
        if cube_red.dims.size:
            sub_est = tuple(self._estado[d] for d in cube_red.dims)
            p1 = float(cube_red.data[seleccionar_subestado(sub_est)])
        else:
            p1 = float(cube_red.data)
        return 1.0 - p1

    def _contrib_futuro(self, pos_fut: int, mecanismo_dims: np.ndarray) -> float:
        """|dist_sub[i] - dist_part[i]| con caché (pos, frozenset(mecanismo))."""
        clave = (pos_fut, frozenset(int(d) for d in mecanismo_dims))
        cacheado = self._cache_contrib.get(clave)
        if cacheado is not None:
            return cacheado
        val = abs(
            float(self.sia_dists_marginales[pos_fut])
            - self._marginal_de(pos_fut, mecanismo_dims)
        )
        self._cache_contrib[clave] = val
        return val

    # ------------------------------------------------------------------ #
    # Evaluación (suma separable por futuro, sin k_partir)                 #
    # ------------------------------------------------------------------ #

    def _evaluar_asignacion(self, asignacion: dict[int, int]) -> float:
        total = 0.0
        for nivel in range(self._n_pre, self._M):  # niveles de futuros (p..p+f-1)
            pos_fut = self._orden[nivel][1]
            g = asignacion[nivel]
            total += self._contrib_futuro(pos_fut, self._mecanismo(asignacion, g))
        return total

    def _dist_particion(self, asignacion: dict[int, int]) -> np.ndarray:
        out = np.empty(self._n_fut, dtype=np.float64)
        for nivel in range(self._n_pre, self._M):
            pos_fut = self._orden[nivel][1]
            g = asignacion[nivel]
            out[pos_fut] = self._marginal_de(pos_fut, self._mecanismo(asignacion, g))
        return out

    def _construir_afinidad(self) -> np.ndarray:
        s = self._sens
        diff = np.abs(s[:, None] - s[None, :])
        return 1.0 / (1.0 + diff)

    # ------------------------------------------------------------------ #
    # Heurística (greedy + búsqueda local + multi-start)                   #
    # ------------------------------------------------------------------ #

    def _heuristico_stream(self, k, A, n_starts, max_iter, timeout):
        mejor = self._local_search(self._greedy(k, A), k, max_iter)
        mejor_emd = self._evaluar_asignacion(mejor)
        for s in range(1, n_starts):
            if timeout is not None and (time.time() - self._t0) >= timeout:
                break
            cand = self._local_search(self._greedy_perturbado(k, A, s), k, max_iter)
            e = self._evaluar_asignacion(cand)
            if e < mejor_emd - 1e-12:
                mejor, mejor_emd = cand, e
        return mejor

    def _greedy_perturbado(self, k: int, A: np.ndarray, seed: int) -> dict[int, int]:
        """Semilla robusta con orden de futuros barajado y presentes aleatorios
        (multi-start). Válida para cualquier p, f, 2<=k<=p+f."""
        rng = np.random.default_rng(seed)
        orden_fut = list(range(self._n_fut))
        rng.shuffle(orden_fut)
        verts = [("F", pos) for pos in orden_fut] + [
            ("P", pos) for pos in range(self._n_pre)
        ]
        asig_vk: dict[tuple[str, int], int] = {}
        for idx, (rol, pos) in enumerate(verts):
            if idx < k:
                g = idx
            elif rol == "F":
                afin = [0.0] * k
                for (r2, p2), g2 in asig_vk.items():
                    if r2 == "F":
                        afin[g2] += float(A[pos][p2])
                g = int(np.argmax(afin))
            else:
                g = int(rng.integers(0, k))  # presentes aleatorios (perturbación)
            asig_vk[(rol, pos)] = g
        return {nivel: asig_vk[v] for nivel, v in enumerate(self._orden)}
