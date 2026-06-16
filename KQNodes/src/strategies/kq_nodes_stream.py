"""KQNodesStream — KQNodes heurístico por *streaming* de columnas, para N grande.

Permite analizar redes donde construir el ``System`` completo (N n-cubos de 2^N en
``src/models/core/system.py``) es inviable en memoria (N≈18..25 → varios GB → OOM).
En vez de materializar el sistema:

  - Lee la TPM desde un ``ColumnStore`` (memmap (N, 2^N)); solo una columna en RAM.
  - Evalúa cada k-partición **futuro a futuro** (la EMD-Efecto es separable),
    reutilizando ``NCube.marginalizar`` sobre la columna en streaming, con caché.
  - Afinidad **aproximada** (media por columna) en una pasada → solo siembra el greedy.
  - Búsqueda **heurística** (greedy Welsh-Powell + búsqueda local move/swap +
    multi-start). NO usa Queyranne ni B&B. ``status='heuristic'`` siempre.

Es el respaldo de escalabilidad de KQNodes: minimiza el **mismo** objetivo δ_k que la
ruta exacta (``kq_nodes.py``), pero sin garantía de óptimo. Para N pequeño coincide con
la solución exacta (ver ``tests/test_kq_nodes_stream.py``).

Supuesto: analiza el sistema **sin condicionamiento de fondo** (condición = todos 1);
``alcance`` y ``mecanismo`` seleccionan el subsistema (igual que KGeoMIPStream y el
escenario de "analizar 25 nodos"). Las distribuciones se calculan con la **misma
convención que** ``System.distribucion_marginal`` (probabilidad indexada en el estado
inicial), de modo que la pérdida y las distribuciones son comparables con la vía exacta.

Hereda de ``KQNodes`` para satisfacer la interfaz ``SIA`` y reutilizar utilidades, pero
**no** construye el ``System`` ni toca las rutas exactas (k=2 Queyranne, voraz, B&B).
"""
from __future__ import annotations

import time

import numpy as np

from src.constants.base import ACTUAL, EFFECT
from src.constants.models import KQNODES_LABEL, KQNODES_STRAREGY_TAG
from src.funcs.format import fmt_k_particion_q
from src.funcs.iit import seleccionar_estado
from src.funcs.tpm_store import ColumnStore
from src.middlewares.slogger import SafeLogger
from src.models.core.ncube import NCube
from src.models.core.solution import Solution
from src.strategies.kq_nodes import KQNodes


class KQNodesStream(KQNodes):
    """Variante heurística de KQNodes por streaming de columnas (no construye System)."""

    def __init__(self, tpm: np.ndarray | None = None):
        """Inicialización ligera: NO llama a ``KQNodes.__init__`` (que construiría
        sesión de profiling y dependería de la TPM completa). Solo prepara el logger;
        todo el estado de ejecución se fija en ``aplicar_estrategia_stream``."""
        self.tpm = tpm
        self.logger = SafeLogger(KQNODES_STRAREGY_TAG)
        self.sia_logger = self.logger

    # ------------------------------------------------------------------ #
    # Punto de entrada                                                     #
    # ------------------------------------------------------------------ #

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
        el ``System`` completo.

        Args:
            store: ColumnStore con la TPM (N, 2^N).
            estado: estado inicial binario del sistema completo (longitud N).
            alcance: bits del alcance/futuro (1 = se conserva ese futuro).
            mecanismo: bits del mecanismo/presente (1 = se conserva ese presente).
            k: número de partes (2 <= k <= p+f, con p presentes y f futuros).
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

        futuros = [i for i, b in enumerate(alcance) if b == "1"]      # f futuros
        presentes = [i for i, b in enumerate(mecanismo) if b == "1"]  # p presentes
        f, p = len(futuros), len(presentes)
        if k < 2:
            raise ValueError(f"k debe ser >= 2, recibido {k}")
        if k > f + p:
            raise ValueError(
                f"k={k} supera el nº de vértices del subsistema (p+f={f + p})"
            )

        # --- Estado para el streaming (admite p != f) ---
        self._store = store
        self._estado = np.array([int(b) for b in estado], dtype=np.int8)
        self._dims_all = np.arange(N, dtype=np.int8)
        self._n_fut = f
        self._n_pre = p
        self._M = p + f
        self._futuros = futuros        # índices reales de nodo futuro
        self._presentes = presentes    # índices reales de dimensión presente
        self._orden = [("P", pos) for pos in range(p)] + [
            ("F", pos) for pos in range(f)
        ]
        self._cache_contrib = {}

        # dist_sub[pos] = marginal del futuro pos conservando TODOS los presentes
        # del subsistema (mecanismo completo). Una pasada por columna.
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

        grupos = self._asignacion_a_kparticion(mejor)
        emd = self._evaluar_asignacion(mejor)
        dist = self._dist_particion(mejor)

        solucion = Solution(
            estrategia=f"{KQNODES_LABEL}-K{k} [heuristic]",
            perdida=emd,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            tiempo_total=time.time() - self._t0,
            particion=fmt_k_particion_q(grupos),
            quiere_hablar=False,
        )
        solucion.modo_ejecucion = "stream"
        solucion.status = "heuristic"
        solucion.nodos_explorados = 0
        return solucion

    # ------------------------------------------------------------------ #
    # Streaming de columnas                                                #
    # ------------------------------------------------------------------ #

    def _marginal_de(self, pos_fut: int, keep_dims: np.ndarray) -> float:
        """Marginal del futuro pos_fut conservando solo ``keep_dims``, seleccionado
        en el estado inicial. Lee una sola columna del store y marginaliza el resto.

        Devuelve la probabilidad indexada (misma convención que
        ``System.distribucion_marginal``), no su complemento."""
        nodo = self._futuros[pos_fut]
        cube = NCube(
            indice=nodo,
            dims=self._dims_all,
            data=self._store.columna_ncubo(nodo),
        )
        cube_red = cube.marginalizar(np.setdiff1d(self._dims_all, keep_dims))
        if cube_red.dims.size:
            sub_est = tuple(self._estado[d] for d in cube_red.dims)
            return float(cube_red.data[seleccionar_estado(sub_est)])
        return float(cube_red.data)

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

    def _mecanismo(self, asignacion: dict[int, int], g: int) -> np.ndarray:
        """Dimensiones presentes asignadas a la parte g (mecanismo de esa parte)."""
        dims = [
            self._presentes[self._orden[nivel][1]]
            for nivel in range(self._n_pre)
            if asignacion.get(nivel) == g
        ]
        return np.array(dims, dtype=np.int8)

    def _asignacion_a_kparticion(
        self, asignacion: dict[int, int]
    ) -> list[list[tuple[int, int]]]:
        """{nivel: id_parte} → lista de grupos de nodos {(tiempo, idx)}.

        Cada presente aporta (ACTUAL, dim_presente) y cada futuro (EFFECT, idx_ncubo)
        a su parte. Las partes vacías se descartan (la búsqueda local nunca vacía
        una parte, así que no deberían ocurrir)."""
        k = max(asignacion.values()) + 1
        grupos: list[list[tuple[int, int]]] = [[] for _ in range(k)]
        for nivel, (rol, pos) in enumerate(self._orden):
            g = asignacion[nivel]
            if rol == "F":
                grupos[g].append((EFFECT, int(self._futuros[pos])))
            else:
                grupos[g].append((ACTUAL, int(self._presentes[pos])))
        return [grupo for grupo in grupos if grupo]

    # ------------------------------------------------------------------ #
    # Afinidad aproximada (sobre la actividad media por columna)           #
    # ------------------------------------------------------------------ #

    def _construir_afinidad(self) -> np.ndarray:
        s = self._sens
        diff = np.abs(s[:, None] - s[None, :])
        return 1.0 / (1.0 + diff)

    # ------------------------------------------------------------------ #
    # Heurística: greedy + búsqueda local (move/swap) + multi-start        #
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

    def _greedy(self, k: int, A: np.ndarray) -> dict[int, int]:
        """Semilla voraz (Welsh-Powell) sobre los p+f vértices bipartitos, válida
        para cualquier p, f y 2<=k<=p+f. Los primeros k vértices abren las k partes;
        el resto: futuros por mayor afinidad, presentes por round-robin (la búsqueda
        local los reubica)."""
        orden_fut = np.argsort(A.sum(axis=1))[::-1].tolist()
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
                g = idx % k
            asig_vk[(rol, pos)] = g
        return {nivel: asig_vk[v] for nivel, v in enumerate(self._orden)}

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

    def _local_search(
        self, asignacion: dict[int, int], k: int, max_iter: int = 2000
    ) -> dict[int, int]:
        """Hill-climbing de primera mejora sobre los p+f vértices:
          - move: reasignar un vértice a otra parte (sin vaciar ninguna parte).
          - swap: intercambiar las partes de dos vértices de partes distintas.
        Acepta solo si δ_k baja. Para en óptimo local, al agotar ``max_iter`` o al
        vencer el ``_deadline``. Aprovecha la caché de evaluación por futuro."""
        mejor = dict(asignacion)
        mejor_emd = self._evaluar_asignacion(mejor)
        M = self._M
        it = 0
        mejoro = True
        while mejoro and it < max_iter:
            if self._deadline is not None and time.time() >= self._deadline:
                break
            mejoro = False

            # --- moves ---
            for v in range(M):
                # Deadline por-candidato: un solo barrido en N grande puede exceder
                # el timeout; sin este chequeo el límite de tiempo no se respeta.
                if self._deadline is not None and time.time() >= self._deadline:
                    return mejor
                g_actual = mejor[v]
                for g in range(k):
                    if g == g_actual:
                        continue
                    cand = dict(mejor)
                    cand[v] = g
                    if len(set(cand.values())) != k:  # no vaciar una parte
                        continue
                    it += 1
                    e = self._evaluar_asignacion(cand)
                    if e < mejor_emd - 1e-12:
                        mejor, mejor_emd, mejoro = cand, e, True
                        break
                if mejoro:
                    break
            if mejoro:
                continue

            # --- swaps ---
            for a in range(M):
                if self._deadline is not None and time.time() >= self._deadline:
                    return mejor
                for b in range(a + 1, M):
                    if mejor[a] == mejor[b]:
                        continue
                    cand = dict(mejor)
                    cand[a], cand[b] = mejor[b], mejor[a]
                    it += 1
                    e = self._evaluar_asignacion(cand)
                    if e < mejor_emd - 1e-12:
                        mejor, mejor_emd, mejoro = cand, e, True
                        break
                if mejoro:
                    break

        return mejor
