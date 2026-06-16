"""KGeoMIP — extensión de GeoMIP a k-particiones (k >= 2).

Estrategia: grafo de afinidad sobre la tabla de costos heredada de GeoMIP +
Branch & Bound best-first con semilla voraz (Welsh-Powell), más un modo
heurístico (greedy + búsqueda local) para sistemas donde el B&B es intratable.

Convención de nomenclatura del proyecto K-QGMIP: la clase principal se llama
`KGeoMIP` ('K' = k-particiones), distinguiéndola de la implementación original
de bi-particiones `GeometricSIA`.

Modelo de partición
-------------------
Se parte el conjunto **bipartito** de 2N vértices del subsistema: N nodos
presentes ``(ACTUAL, d)`` y N nodos futuros ``(EFECTO, j)``, asignados de forma
independiente a k partes. Este es el mismo modelo de vértices que usan GeoMIP y
QNodes originales, por lo que para k=2 KGeoMIP reproduce su bipartición.

La pérdida δₖ = EMD-Efecto es separable por nodo futuro:

    δₖ = Σ_i |dist_sub[i] - dist_part[i]|

donde dist_part[i] depende únicamente de los nodos presentes que comparten la
parte del futuro i. Esto se explota para construir cotas inferiores admisibles
en la fase de asignación de futuros.

Modos de ejecución (parámetro ``mode``)
---------------------------------------
- ``exact``: B&B sin tope de nodos (óptimo global si termina). Admite ``timeout``.
- ``bnb``:   B&B con tope de nodos y/o ``timeout`` (anytime: devuelve la mejor
             solución hallada, con estado ``optimal``/``capped``/``timeout``).
- ``heuristic``: greedy (Welsh-Powell) + búsqueda local; rápido, sin garantía de
             óptimo (estado ``heuristic``).
- ``auto`` (por defecto): ejecuta B&B acotado (que **certifica el óptimo** cuando
             la búsqueda cabe en el tope de nodos) y, si se trunca (capped/timeout),
             usa la heurística como respaldo quedándose con la mejor solución. Si N
             supera el límite de la tabla 2^N, reporta ``infeasible``.

El resultado se etiqueta siempre con su estado en ``Solution.estrategia`` y en
los atributos ``Solution.modo_ejecucion``, ``Solution.status`` y
``Solution.nodos_explorados``. Nunca se reporta ``optimal`` sin certificarlo.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np

from src.constants.base import ACTUAL, EFECTO
from src.constants.models import KGEOMIP_LABEL
from src.controllers.strategies.geometric import GeometricSIA
from src.funcs.base import emd_efecto, seleccionar_subestado
from src.funcs.format import fmt_k_particion
from src.models.core.solution import Solution


# Una k-partición: lista de frozensets, cada uno {(tiempo, idx_nodo), ...}
KParticion = list[frozenset[tuple[int, int]]]

# Tope de nodos explorados en B&B (cortafuegos de TIEMPO).
_MAX_NODOS = 300_000

# Tope del tamaño de la frontera (heap) del B&B (cortafuegos de MEMORIA): los
# `push` no están acotados por _MAX_NODOS (que cuenta `pop`), así que en
# subsistemas grandes el heap crecería sin límite → OOM. Al alcanzarlo, el B&B
# se trunca (`capped`) y `auto` cae al respaldo heurístico.
_MAX_HEAP = 50_000

# Cota de N para la cual construir la tabla de costos (2^N estados) es viable.
# Por encima de esto, el modo `auto` reporta `infeasible` (la afinidad, y por
# tanto exact/bnb/heuristic, necesitan la tabla).
_N_MAX_TABLA = 16

# Estados posibles de una solución (honestidad sobre la garantía).
ESTADO_OPTIMO = "optimal"
ESTADO_CAPADO = "capped"
ESTADO_TIMEOUT = "timeout"
ESTADO_HEURISTICO = "heuristic"
ESTADO_INFACTIBLE = "infeasible"


@dataclass
class _NodoBnB:
    """Nodo del árbol de Branch & Bound."""

    lb: float
    nivel: int                 # siguiente vértice a asignar (0..2N-1)
    asignacion: dict[int, int]  # nivel -> id de parte
    phi_parcial: float          # suma de contribuciones de futuros ya fijados


class KGeoMIP(GeometricSIA):
    """
    Encuentra la k-Partición de Mínima Información (k-MIP) extendiendo GeoMIP.

    Reutiliza de `GeometricSIA` la infraestructura de N-Cubos y la **tabla de
    costos de transiciones** (calculada una sola vez, independientemente de k).
    Sobre ella construye un grafo de afinidad y resuelve según el modo elegido
    (exact / bnb / heuristic / auto). Ver el docstring del módulo.

    Limitaciones declaradas:
      - El B&B no tiene cota inferior informativa en la fase de presentes; la
        memoización (cachés) lo hace viable en la práctica hasta ~N10/k4 (óptimo
        certificado en segundos). Donde el tope de nodos no basta, `auto` cae al
        respaldo heurístico (sin garantía de óptimo, etiquetado como tal).
      - Admite subsistemas con |presente| (p) != |futuro| (f): parte el conjunto
        bipartito de p+f vértices.
      - Requiere construir la tabla de costos 2^p (límite práctico p≈16).
    """

    def aplicar_estrategia_k(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: int,
        *,
        mode: str = "auto",
        timeout: float | None = None,
        max_nodos: int = _MAX_NODOS,
    ) -> Solution:
        """
        Punto de entrada: halla la k-MIP del subsistema indicado.

        Args:
            condicion: bits de condiciones de fondo (0 = condicionar).
            alcance: bits del alcance/futuro (0 = substraer).
            mecanismo: bits del mecanismo/presente (0 = substraer).
            tpm: Matriz de Probabilidad de Transición del sistema.
            k: número de partes de la partición (2 <= k <= N).
            mode: 'auto' (def.), 'exact', 'bnb' o 'heuristic'.
            timeout: segundos máximos para el B&B (None = sin límite).
            max_nodos: tope de nodos del B&B en modo 'bnb'.

        Returns:
            Solution con la pérdida δₖ, la distribución de la partición, la
            k-partición formateada y la telemetría (modo_ejecucion, status,
            nodos_explorados).

        Raises:
            ValueError: si k < 2, k > p+f o `mode` desconocido.
        """
        if k < 2:
            raise ValueError(f"k debe ser >= 2, recibido {k}")
        if mode not in ("auto", "exact", "bnb", "heuristic"):
            raise ValueError(f"mode desconocido: {mode!r}")

        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)
        f = len(self.sia_subsistema.indices_ncubos)  # nodos futuros del subsistema
        p = len(self.sia_subsistema.dims_ncubos)     # nodos presentes del subsistema

        # Se parte el conjunto bipartito de p+f vértices (admite p != f).
        if k > f + p:
            raise ValueError(f"k={k} supera el nº de vértices del subsistema (p+f={f + p})")

        # Estado por llamada (permite reutilizar la instancia entre subsistemas
        # sin acumular: tabla de costos, cachés y deadline se reinician aquí).
        self.tabla_transiciones = {}
        self._cache_contrib: dict[tuple[int, frozenset], float] = {}
        self._cache_eval: dict[frozenset, tuple[float, np.ndarray]] = {}
        self._deadline = (self.sia_tiempo_inicio + timeout) if timeout else None

        # La tabla de costos se construye sobre el hipercubo de los p presentes
        # (2^p estados). Si p es demasiado grande, es infactible por memoria.
        if p > _N_MAX_TABLA:
            return self._solucion_infactible(k, p)

        # --- Conteos del subsistema bipartito ---
        self._n_fut = f
        self._n_pre = p
        self._M = p + f
        self._futuros = list(self.sia_subsistema.indices_ncubos)
        self._presentes = list(self.sia_subsistema.dims_ncubos)
        # Orden de vértices: presentes (0..p-1), luego futuros (p..p+f-1).
        self._orden = [("P", pos) for pos in range(p)] + [("F", pos) for pos in range(f)]

        # --- Infraestructura heredada: tabla de costos (se calcula 1 vez) ---
        self._flat_data = [cube.data.ravel() for cube in self.sia_subsistema.ncubos]
        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial
        self.idx_ncubos = list(range(f))
        self.caminos = {0: [self.estado_inicial.tolist()]}
        self.tabla_transiciones[
            (tuple(self.caminos[0][0]), tuple(self.caminos[0][0]))
        ] = [0.0] * f
        for nivel in range(1, len(self.estado_inicial) + 1):
            self.calcular_costos_nivel(self.estado_final, nivel)

        # --- Búsqueda de la k-partición según el modo ---
        A = self._construir_afinidad()
        if mode == "heuristic":
            mejor = self._heuristico(k, A)
            status, nodos, modo = ESTADO_HEURISTICO, 0, "heuristic"
        elif mode == "exact":
            mejor, status, nodos = self._bnb(k, A, max_nodos=None, timeout=timeout)
            modo = "exact"
        elif mode == "bnb":
            mejor, status, nodos = self._bnb(k, A, max_nodos=max_nodos, timeout=timeout)
            modo = "bnb"
        else:  # auto: B&B acotado (certifica óptimo si cabe) + respaldo heurístico
            modo = "auto"
            mejor, status, nodos = self._bnb(k, A, max_nodos=max_nodos, timeout=timeout)
            if status != ESTADO_OPTIMO:
                heur = self._heuristico(k, A)
                if self._evaluar_asignacion(heur) < self._evaluar_asignacion(mejor) - 1e-12:
                    mejor, status = heur, ESTADO_HEURISTICO

        kpart = self._asignacion_a_kparticion(mejor)
        emd, dist = self._evaluar_kparticion(kpart)

        solucion = Solution(
            estrategia=f"{KGEOMIP_LABEL}-K{k} [{status}]",
            perdida=emd,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_k_particion(kpart),
        )
        solucion.modo_ejecucion = modo
        solucion.status = status
        solucion.nodos_explorados = nodos
        return solucion

    # ------------------------------------------------------------------ #
    # Infactibilidad                                                       #
    # ------------------------------------------------------------------ #

    def _solucion_infactible(self, k: int, p: int) -> Solution:
        """Solución etiquetada como infactible (no se construye la tabla 2^p)."""
        solucion = Solution(
            estrategia=f"{KGEOMIP_LABEL}-K{k} [{ESTADO_INFACTIBLE}]",
            perdida=float("inf"),
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.sia_dists_marginales,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=f"INFEASIBLE: p={p} presentes excede el límite de la tabla 2^p (k={k}).",
        )
        solucion.modo_ejecucion = ESTADO_INFACTIBLE
        solucion.status = ESTADO_INFACTIBLE
        solucion.nodos_explorados = 0
        return solucion

    # ------------------------------------------------------------------ #
    # Grafo de afinidad (sobre los costos de los nodos futuros)           #
    # ------------------------------------------------------------------ #

    def _construir_afinidad(self) -> np.ndarray:
        """A[i][j] = 1 / (1 + |cost_i - cost_j|). Alta afinidad → misma parte."""
        key = (tuple(self.caminos[0][0]), tuple(self.estado_final.tolist()))
        costos = np.array(self.tabla_transiciones[key], dtype=np.float64)
        diff = np.abs(costos[:, None] - costos[None, :])
        return 1.0 / (1.0 + diff)

    # ------------------------------------------------------------------ #
    # Solución inicial voraz (Welsh-Powell sobre A)                        #
    # ------------------------------------------------------------------ #

    def _greedy(self, k: int, A: np.ndarray) -> dict[int, int]:
        """
        Semilla voraz (Welsh-Powell) sobre los p+f vértices bipartitos, válida para
        cualquier p, f y 2<=k<=p+f:
          - Los primeros k vértices (futuros por afinidad, luego presentes) abren las
            k partes.
          - El resto: futuros por mayor afinidad a las partes ya formadas; presentes
            por round-robin (la búsqueda local los reubica después).
        Devuelve una asignación (nivel -> parte) con las k partes no vacías.
        """
        orden_fut = np.argsort(A.sum(axis=1))[::-1].tolist()  # f futuros por afinidad
        verts = [("F", pos) for pos in orden_fut] + [
            ("P", pos) for pos in range(self._n_pre)
        ]
        asig_vk: dict[tuple[str, int], int] = {}
        for idx, (rol, pos) in enumerate(verts):
            if idx < k:
                g = idx  # abrir las k partes
            elif rol == "F":
                afin = [0.0] * k
                for (r2, p2), g2 in asig_vk.items():
                    if r2 == "F":
                        afin[g2] += float(A[pos][p2])
                g = int(np.argmax(afin))
            else:
                g = idx % k  # presentes: round-robin (refinado por búsqueda local)
            asig_vk[(rol, pos)] = g
        return {nivel: asig_vk[v] for nivel, v in enumerate(self._orden)}

    # ------------------------------------------------------------------ #
    # Heurística: greedy + búsqueda local (move/swap)                      #
    # ------------------------------------------------------------------ #

    def _heuristico(self, k: int, A: np.ndarray) -> dict[int, int]:
        """Semilla voraz + búsqueda local. Rápido, sin garantía de óptimo."""
        return self._local_search(self._greedy(k, A), k)

    def _local_search(
        self, asignacion: dict[int, int], k: int, max_iter: int = 2000
    ) -> dict[int, int]:
        """
        Hill-climbing de primera mejora sobre los p+f vértices:
          - move: reasignar un vértice a otra parte (sin vaciar ninguna parte).
          - swap: intercambiar las partes de dos vértices de partes distintas.
        Acepta solo si δₖ baja. Para en óptimo local o al agotar `max_iter`.
        Aprovecha la caché de evaluación de particiones.
        """
        mejor = dict(asignacion)
        mejor_emd = self._evaluar_asignacion(mejor)
        M = self._M
        it = 0
        mejoro = True
        while mejoro and it < max_iter:
            # Cortafuegos de tiempo opcional (lo fija el modo stream).
            if getattr(self, "_deadline", None) and time.time() >= self._deadline:
                break
            mejoro = False

            # --- moves ---
            for v in range(M):
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

    # ------------------------------------------------------------------ #
    # Branch & Bound (anytime: timeout + tope de nodos + telemetría)       #
    # ------------------------------------------------------------------ #

    def _bnb(
        self,
        k: int,
        A: np.ndarray,
        max_nodos: int | None = _MAX_NODOS,
        timeout: float | None = None,
    ) -> tuple[dict[int, int], str, int]:
        """
        B&B best-first sobre 2N vértices, cola de prioridad por cota inferior.

        Ruptura de simetría: la parte g solo se abre si 0..g-1 ya tienen al menos
        un vértice (orden canónico de etiquetas).

        Returns:
            (mejor_asignacion, status, nodos_explorados) donde status es
            'optimal' (la cola se vació sin truncar → óptimo certificado),
            'capped' (se alcanzó max_nodos) o 'timeout' (se agotó el tiempo).
        """
        M = self._M
        limite = float("inf") if max_nodos is None else max_nodos
        t0 = self.sia_tiempo_inicio

        mejor = self._greedy(k, A)
        cota = self._evaluar_asignacion(mejor)

        raiz = _NodoBnB(lb=0.0, nivel=0, asignacion={}, phi_parcial=0.0)
        contador = 0
        cola: list[tuple[float, int, _NodoBnB]] = [(0.0, 0, raiz)]
        nodos = 0
        status = ESTADO_OPTIMO

        while cola:
            if nodos >= limite:
                status = ESTADO_CAPADO
                break
            if len(cola) >= _MAX_HEAP:  # cortafuegos de memoria
                status = ESTADO_CAPADO
                break
            if timeout is not None and (time.time() - t0) >= timeout:
                status = ESTADO_TIMEOUT
                break

            _, _, nodo = heapq.heappop(cola)
            nodos += 1

            if nodo.lb >= cota:
                continue

            if nodo.nivel == M:
                phi = self._evaluar_asignacion(nodo.asignacion)
                if phi < cota:
                    cota, mejor = phi, dict(nodo.asignacion)
                continue

            nivel = nodo.nivel
            rol, pos = self._orden[nivel]
            for g in self._grupos_disponibles(nodo.asignacion, k):
                nueva = {**nodo.asignacion, nivel: g}
                if not self._factible(nueva, k, M):
                    continue

                # Solo los futuros aportan a δₖ, y solo cuando ya están fijados
                # todos los presentes (presentes en niveles 0..N-1).
                if rol == "F":
                    mec = self._mecanismo(nueva, g)
                    contrib = self._contrib_futuro(pos, mec)
                else:
                    contrib = 0.0
                phi_p = nodo.phi_parcial + contrib
                if phi_p >= cota:
                    continue

                lb = self._cota_inferior(nivel + 1, nueva, phi_p, k)
                if lb >= cota:
                    continue

                contador += 1
                heapq.heappush(
                    cola,
                    (
                        lb,
                        contador,
                        _NodoBnB(
                            lb=lb,
                            nivel=nivel + 1,
                            asignacion=nueva,
                            phi_parcial=phi_p,
                        ),
                    ),
                )

        return mejor, status, nodos

    # ------------------------------------------------------------------ #
    # Utilidades del B&B                                                   #
    # ------------------------------------------------------------------ #

    def _grupos_disponibles(self, asignacion: dict[int, int], k: int) -> list[int]:
        """Partes ya usadas + (si quedan) la siguiente parte vacía. Ruptura de simetría."""
        usados = set(asignacion.values())
        sig_nuevo = len(usados)
        grupos = list(usados)
        if sig_nuevo < k:
            grupos.append(sig_nuevo)
        return grupos

    def _factible(self, asignacion: dict[int, int], k: int, M: int) -> bool:
        """Los vértices restantes deben bastar para llenar las partes vacías."""
        vacios = k - len(set(asignacion.values()))
        restantes = M - len(asignacion)
        return vacios <= restantes

    def _mecanismo(self, asignacion: dict[int, int], g: int) -> np.ndarray:
        """Dimensiones presentes asignadas a la parte g (mecanismo de esa parte)."""
        dims = [
            self._presentes[self._orden[nivel][1]]
            for nivel in range(self._n_pre)
            if asignacion.get(nivel) == g
        ]
        return np.array(dims, dtype=np.int8)

    def _contrib_futuro(self, pos_fut: int, mecanismo_dims: np.ndarray) -> float:
        """
        Contribución exacta del nodo futuro pos_fut a δₖ dada la mecanismo de su
        parte: |dist_sub[i] - dist_part[i]|, donde dist_part[i] resulta de
        marginalizar el n-cubo i conservando solo las dims de su parte.

        Memoizado por clave inmutable (pos_fut, frozenset(mecanismo)): la
        marginalización depende solo del *conjunto* de dims conservadas.
        """
        clave = (pos_fut, frozenset(int(d) for d in mecanismo_dims))
        cacheado = self._cache_contrib.get(clave)
        if cacheado is not None:
            return cacheado

        cube = self.sia_subsistema.ncubos[pos_fut]
        cube_red = cube.marginalizar(np.setdiff1d(cube.dims, mecanismo_dims))
        if cube_red.dims.size:
            sub_est = tuple(
                self.sia_subsistema.estado_inicial[d] for d in cube_red.dims
            )
            p1 = float(cube_red.data[seleccionar_subestado(sub_est)])
        else:
            p1 = float(cube_red.data)
        valor = abs(float(self.sia_dists_marginales[pos_fut]) - (1.0 - p1))
        self._cache_contrib[clave] = valor
        return valor

    def _cota_inferior(
        self,
        nivel: int,
        asignacion: dict[int, int],
        phi_parcial: float,
        k: int,
    ) -> float:
        """
        Cota inferior:
          - Fase de presentes (nivel < p): aún no se conocen las mecanismos
            finales ⇒ lb = phi_parcial (= 0, sin futuros fijados).
          - Fase de futuros (nivel >= p): todas las mecanismos son finales ⇒
            por cada futuro no asignado se suma su contribución mínima sobre las
            k partes. Es **admisible** (min ≤ valor real), porque δₖ es separable
            por futuro dada la asignación de presentes.
        """
        if nivel < self._n_pre:
            return phi_parcial

        mecs = [self._mecanismo(asignacion, g) for g in range(k)]
        lb = phi_parcial
        for niv in range(max(self._n_pre, nivel), self._M):
            pos_fut = self._orden[niv][1]
            lb += min(self._contrib_futuro(pos_fut, mecs[g]) for g in range(k))
        return lb

    # ------------------------------------------------------------------ #
    # Evaluación de particiones                                            #
    # ------------------------------------------------------------------ #

    def _evaluar_asignacion(self, asignacion: dict[int, int]) -> float:
        emd, _ = self._evaluar_kparticion(self._asignacion_a_kparticion(asignacion))
        return emd

    def _evaluar_kparticion(self, kpart: KParticion) -> tuple[float, np.ndarray]:
        """
        Pérdida δₖ exacta de una k-partición vía System.k_partir + EMD-Efecto.

        Memoizado por clave `frozenset(kpart)` (invariante al reetiquetado de
        partes; δₖ también lo es).
        """
        clave = frozenset(kpart)
        cacheado = self._cache_eval.get(clave)
        if cacheado is not None:
            return cacheado

        grupos = [
            (
                np.array([idx for (t, idx) in gr if t == EFECTO], dtype=np.int8),
                np.array([idx for (t, idx) in gr if t == ACTUAL], dtype=np.int8),
            )
            for gr in kpart
        ]
        dist = self.sia_subsistema.k_partir(grupos).distribucion_marginal()
        emd = emd_efecto(dist, self.sia_dists_marginales)
        self._cache_eval[clave] = (emd, dist)
        return emd, dist

    def _asignacion_a_kparticion(self, asignacion: dict[int, int]) -> KParticion:
        """
        {nivel: id_parte} → lista de frozensets {(tiempo, idx_nodo)}.

        Cada presente (nivel < N) aporta (ACTUAL, dim) y cada futuro
        (nivel >= N) aporta (EFECTO, idx_ncubo) a su parte. Las partes vacías se
        descartan (no ocurren en una hoja factible del B&B).
        """
        k = max(asignacion.values()) + 1
        grupos: list[set[tuple[int, int]]] = [set() for _ in range(k)]
        for nivel, (rol, pos) in enumerate(self._orden):
            g = asignacion[nivel]
            if rol == "F":
                grupos[g].add((EFECTO, int(self._futuros[pos])))
            else:
                grupos[g].add((ACTUAL, int(self._presentes[pos])))
        return [frozenset(s) for s in grupos if s]
