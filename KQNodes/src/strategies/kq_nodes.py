import time
from typing import Union
import numpy as np
from src.middlewares.slogger import SafeLogger
from src.funcs.iit import emd_efecto, ABECEDARY
from src.middlewares.profile import gestor_perfilado, profile
from src.funcs.format import fmt_biparticion_q
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.models.core.system import System
from src.constants.models import (
    QNODES_ANALYSIS_TAG,
    QNODES_STRAREGY_TAG,
)
from src.constants.base import (
    COLS_IDX,
    INT_ZERO,
    TYPE_TAG,
    NET_LABEL,
    INFTY_POS,
    LAST_IDX,
    EFFECT,
    ACTUAL,
)
from src.models.base.application import aplicacion

_BNB_EXACT_THRESHOLD = 10   # n ≤ 10  → B&B 
_BNB_DEFAULT_TIMEOUT = 90.0 # segundos máximos para B&B con time-limit
 
 
class KQNodes(SIA):
    """
    Estrategia KQNodes: extensión de QNodes a k-particiones (k en [2, 5]).

    Para k=2 reproduce el algoritmo QNodes original (minimización submodular tipo
    Queyranne). Para k>=3 combina una semilla voraz (siembra por distancia y
    asignación submodular), refinamiento local (mover vértices entre grupos) y un
    Branch & Bound con poda y límite de tiempo. Hereda de `SIA` y mide la pérdida
    con la EMD-Efecto entre el subsistema y la reconstrucción k-partita.
    """

    def __init__(self, tpm: np.ndarray):
        """Inicializa la estrategia para una TPM dada: sesión de profiling, logger,
        memorias de submodularidad y estado interno del Branch & Bound."""
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self.m: int
        self.n: int
        self.tiempos: tuple[np.ndarray, np.ndarray]
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set[tuple]
        self.clave_submodular = [], []
        self.memoria_delta: dict = {}
        self.memoria_grupo_candidato: dict = {}
 
        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray
 
        self.logger = SafeLogger(QNODES_STRAREGY_TAG)
 

        self._bnb_mejor_perdida: float = INFTY_POS
        self._bnb_mejor_grupos: list[list] | None = None
        self._bnb_mejor_dist: np.ndarray | None = None
        self._bnb_timeout: float = _BNB_DEFAULT_TIMEOUT
        self._bnb_inicio: float = 0.0
        self._bnb_podas: int = 0  
    
    def _validar_kparticion(self, grupos, vertices, k: int):
        """Valida que `grupos` sea una k-partición correcta de `vertices`: k grupos
        no vacíos, nodos (tiempo, idx) bien formados, sin repetidos y cubriendo
        exactamente los vértices. Lanza ValueError si no se cumple."""
        if not grupos:
            raise ValueError("La k-partición está vacía.")

        if len(grupos) != k:
            raise ValueError(
                f"La partición debe tener exactamente {k} grupos, "
                f"pero tiene {len(grupos)}."
            )

        nodos_vistos = []

        for i, grupo in enumerate(grupos):
            if not grupo:
                raise ValueError(f"El grupo {i} está vacío.")

            for nodo in grupo:
                if not (
                    isinstance(nodo, (tuple, list))
                    and len(nodo) == 2
                    and isinstance(nodo[0], (int, np.integer))
                    and isinstance(nodo[1], (int, np.integer))
                ):
                    raise ValueError(f"Nodo malformado en grupo {i}: {nodo}")

                nodos_vistos.append((int(nodo[0]), int(nodo[1])))

        if len(nodos_vistos) != len(set(nodos_vistos)):
            raise ValueError("Hay nodos repetidos en la k-partición.")

        vertices_set = {(int(t), int(i)) for t, i in vertices}
        nodos_set = set(nodos_vistos)

        if nodos_set != vertices_set:
            faltantes = vertices_set - nodos_set
            extras = nodos_set - vertices_set
            raise ValueError(
                f"La k-partición no cubre exactamente los vértices. "
                f"Faltantes: {faltantes}. Extras: {extras}."
            )

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int = 2,
    ) -> Solution:
        """
        Punto de entrada: halla la k-partición de mínima información del subsistema.

        Args:
            estado_inicial: estado inicial del sistema (cadena binaria).
            condicion: condiciones de fondo (0 = condicionar).
            alcance: bits del alcance/futuro (0 = substraer).
            mecanismo: bits del mecanismo/presente (0 = substraer).
            k: número de grupos, en [2, 5]. k=2 usa el QNodes original; k>=3 usa
               semilla voraz + refinamiento local + Branch & Bound.

        Returns:
            Solution con la pérdida (EMD-Efecto), la distribución de la partición y
            la k-partición formateada.
        """
        if not (2 <= k <= 5):
            raise ValueError(f"k debe estar en [2, 5], se recibió k={k}")
 
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)
        self.memoria_delta = {}
        self.memoria_grupo_candidato = {}
 
        futuro = tuple(
            (EFFECT, idx) for idx in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, idx) for idx in self.sia_subsistema.dims_ncubos
        )
 
        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size
        self.indices_alcance   = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        self.tiempos = (
            np.zeros(self.n, dtype=np.int8),
            np.zeros(self.m, dtype=np.int8),
        )
 
        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)
        total_vertices = len(vertices)

        if k > total_vertices:
            raise ValueError(
                f"k={k} no puede ser mayor que el número de vértices "
                f"del subsistema ({total_vertices})."
            )
 
        if k == 2:
            grupos_fin, perdida_fin, dist_fin = self._resolver_k2_qnodes_original(vertices)

        else:
            grupos_voraz, perdida_voraz, dist_voraz = self._voraz_k(vertices, k)

            grupos_ref, perdida_ref, dist_ref = self._refinamiento_local(
                grupos_voraz,
                perdida_voraz,
                dist_voraz
            )
            self._validar_kparticion(grupos_ref, vertices, k)
            sin_timeout = total_vertices <= _BNB_EXACT_THRESHOLD
            timeout = None if sin_timeout else self._bnb_timeout

            grupos_bnb, perdida_bnb, dist_bnb = self._branch_and_bound(
                vertices,
                k,
                upper_bound=perdida_ref,
                grupos_iniciales=grupos_ref,
                dist_inicial=dist_ref,
                timeout=timeout,
            )

            grupos_fin  = grupos_bnb
            perdida_fin = perdida_bnb
            dist_fin    = dist_bnb
    
        self._validar_kparticion(grupos_fin, vertices, k)
        particion_fmt = self._fmt_k_particion(grupos_fin)
 
        return Solution(
            estrategia=f"KQNodes k={k}",
            perdida=perdida_fin,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_fin,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=particion_fmt,
        )
 
    def _resolver_k2_qnodes_original(
        self,
        vertices: list[tuple],
    ) -> tuple[list[list], float, np.ndarray]:
        """Resuelve k=2 con el algoritmo QNodes original (Queyranne): obtiene la
        mejor bipartición (grupo, complemento) con su pérdida y distribución."""
        self.memoria_grupo_candidato = {}

        mip = self._algorithm_qnodes_original(list(vertices))

        grupo_a = list(mip)
        set_a = set(grupo_a)
        grupo_b = [v for v in vertices if v not in set_a]

        if not grupo_a or not grupo_b:
            raise ValueError("QNodes produjo una bipartición inválida.")

        perdida, dist = self.memoria_grupo_candidato[mip]

        return [grupo_a, grupo_b], perdida, dist
    
    def _algorithm_qnodes_original(self, vertices: list[tuple[int, int]]):
        """Algoritmo Q (Queyranne): construye grupos incrementalmente con la función
        submodular y devuelve la clave de la mejor bipartición candidata (memorizada
        en `memoria_grupo_candidato`)."""
        indice_emd = INT_ZERO

        for i in range(len(vertices) - 1):
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = vertices[1:]

            emd_particion_candidata = INFTY_POS
            dist_particion_candidata = None

            for j in range(len(deltas_ciclo) - 1):
                emd_local = INFTY_POS
                indice_mip = None

                for idx_delta in range(len(deltas_ciclo)):
                    emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular(
                        deltas_ciclo[idx_delta],
                        omegas_ciclo,
                    )

                    emd_iteracion = emd_union - emd_delta

                    if emd_delta == INT_ZERO:
                        clave = (
                            tuple(deltas_ciclo[idx_delta])
                            if isinstance(deltas_ciclo[idx_delta], list)
                            else (deltas_ciclo[idx_delta],)
                        )

                        self.memoria_grupo_candidato[clave] = (
                            emd_delta,
                            dist_marginal_delta,
                        )

                        return clave

                    if emd_iteracion < emd_local:
                        emd_local = emd_iteracion
                        indice_mip = idx_delta
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal_delta

                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)

            clave_candidata = tuple(
                deltas_ciclo[LAST_IDX]
                if isinstance(deltas_ciclo[LAST_IDX], list)
                else deltas_ciclo
            )

            self.memoria_grupo_candidato[clave_candidata] = (
                emd_particion_candidata,
                dist_particion_candidata,
            )

            par_candidato = (
                [omegas_ciclo[LAST_IDX]]
                if isinstance(omegas_ciclo[LAST_IDX], tuple)
                else omegas_ciclo[LAST_IDX]
            ) + (
                deltas_ciclo[LAST_IDX]
                if isinstance(deltas_ciclo[LAST_IDX], list)
                else deltas_ciclo
            )

            omegas_ciclo.pop()
            omegas_ciclo.append(par_candidato)

            vertices = omegas_ciclo

        return min(
            self.memoria_grupo_candidato,
            key=lambda clave: self.memoria_grupo_candidato[clave][indice_emd],
        )
 
    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def _voraz_k(
        self,
        vertices: list[tuple],
        k: int,
    ) -> tuple[list[list], float, np.ndarray]:
        """Construcción voraz de k grupos: parte de k semillas y asigna cada vértice
        restante al grupo de menor costo submodular. Devuelve (grupos, pérdida, dist)."""
        grupos = self._inicializar_seeds(vertices, k)

        asignados = {v for grupo in grupos for v in grupo}
        deltas = [v for v in vertices if v not in asignados]

        while deltas:
            mejor_costo = INFTY_POS
            mejor_d_idx = None
            mejor_g_idx = None

            for g_idx, grupo in enumerate(grupos):
                for d_idx, delta in enumerate(deltas):
                    emd_union, emd_delta, _ = self.funcion_submodular(
                        delta,
                        grupo,
                    )

                    costo = emd_union - emd_delta

                    if costo < mejor_costo:
                        mejor_costo = costo
                        mejor_d_idx = d_idx
                        mejor_g_idx = g_idx

            if mejor_d_idx is None or mejor_g_idx is None:
                raise ValueError("No se pudo asignar un delta en _voraz_k.")

            grupos[mejor_g_idx].append(deltas.pop(mejor_d_idx))

        perdida, dist = self._emd_k_particion(grupos)

        clave = self._clave_k(grupos)
        self.memoria_grupo_candidato[clave] = (perdida, dist)

        return grupos, perdida, dist
 
    def _inicializar_seeds(
        self, vertices: list[tuple], k: int
    ) -> list[list[tuple]]:
        """Elige k semillas iniciales lo más separadas posible (máxima distancia de
        Hamming mínima a las ya elegidas); devuelve un grupo por semilla."""
        if k >= len(vertices):
            return [[v] for v in vertices[:k]]
 
        seeds = [vertices[0]]
        candidatos = list(vertices[1:])
 
        for _ in range(k - 1):
            dists = [
                min(self._hamming_vertices(c, s) for s in seeds)
                for c in candidatos
            ]
            mejor = int(np.argmax(dists))
            seeds.append(candidatos.pop(mejor))
 
        return [[s] for s in seeds]
 
    @staticmethod
    def _hamming_vertices(v1: tuple, v2: tuple) -> int:
        """Distancia de Hamming entre dos vértices (tiempo, idx): 0, 1 o 2."""
        t1, i1 = v1
        t2, i2 = v2
        return int(t1 != t2) + int(i1 != i2)
 
    def _refinamiento_local(
        self,
        grupos: list[list],
        perdida_actual: float,
        dist_actual: np.ndarray,
        max_pasadas: int = 10,
    ) -> tuple[list[list], float, np.ndarray]:
        """Búsqueda local: mueve vértices entre grupos mientras baje la pérdida (sin
        vaciar ningún grupo), hasta `max_pasadas` o hasta no mejorar."""
        grupos = [list(g) for g in grupos]

        for _ in range(max_pasadas):
            mejor_grupos = None
            mejor_perdida = perdida_actual
            mejor_dist = dist_actual

            for g_origen_idx, grupo_origen in enumerate(grupos):
                if len(grupo_origen) <= 1:
                    continue

                for vertice in list(grupo_origen):
                    for g_dest_idx in range(len(grupos)):
                        if g_dest_idx == g_origen_idx:
                            continue

                        candidato = [list(g) for g in grupos]

                        if vertice not in candidato[g_origen_idx]:
                            continue

                        candidato[g_origen_idx].remove(vertice)
                        candidato[g_dest_idx].append(vertice)

                        # No aceptar particiones con grupos vacíos
                        if any(len(g) == 0 for g in candidato):
                            continue

                        nueva_perdida, nueva_dist = self._emd_k_particion(candidato)

                        if nueva_perdida < mejor_perdida - 1e-10:
                            mejor_perdida = nueva_perdida
                            mejor_dist = nueva_dist
                            mejor_grupos = candidato

            if mejor_grupos is None:
                break

            grupos = mejor_grupos
            perdida_actual = mejor_perdida
            dist_actual = mejor_dist

        return grupos, perdida_actual, dist_actual
 
    def _branch_and_bound(
        self,
        vertices: list[tuple],
        k: int,
        upper_bound: float,
        grupos_iniciales: list[list],
        dist_inicial: np.ndarray,
        timeout: float | None = None,
    ) -> tuple[list[list], float, np.ndarray]:
        """Branch & Bound sobre las k-particiones: parte de la cota superior de la
        heurística y explora asignaciones podando por cota; respeta un `timeout`
        opcional. Devuelve la mejor (grupos, pérdida, dist) hallada."""
        self._bnb_mejor_perdida = upper_bound
        self._bnb_mejor_grupos  = [list(g) for g in grupos_iniciales]
        self._bnb_mejor_dist    = dist_inicial
        self._bnb_inicio        = time.time()
        self._bnb_timeout       = timeout if timeout is not None else float("inf")
        self._bnb_podas         = 0
 
        grupos_vacios = [[] for _ in range(k)]
        self._bnb_explorar(grupos_vacios, list(vertices), k)
 
        self.logger.debug(
            f"B&B terminado: {self._bnb_podas} podas, "
            f"mejor={self._bnb_mejor_perdida:.6f}"
        )
        return (
            self._bnb_mejor_grupos,
            self._bnb_mejor_perdida,
            self._bnb_mejor_dist,
        )
 
    def _bnb_explorar(
        self,
        grupos: list[list],
        deltas: list[tuple],
        k: int,
    ) -> None:
        """Recursión del B&B: asigna el siguiente vértice a cada grupo factible
        (con ruptura de simetría de grupos vacíos), poda por timeout, por grupos
        vacíos imposibles de llenar y por cota inferior parcial, y actualiza la
        mejor partición al completar una asignación."""
        if time.time() - self._bnb_inicio > self._bnb_timeout:
            return
        
        grupos_vacios = sum(1 for g in grupos if len(g) == 0)
        if len(deltas) < grupos_vacios:
            self._bnb_podas += 1
            return

        if not deltas:
            if grupos_vacios > 0:
                self._bnb_podas += 1
                return

            perdida, dist = self._emd_k_particion(grupos)

            if perdida < self._bnb_mejor_perdida:
                self._bnb_mejor_perdida = perdida
                self._bnb_mejor_grupos  = [list(g) for g in grupos]
                self._bnb_mejor_dist    = dist

            return

        # NOTA: aquí había una poda por "cota inferior" usando
        # `_emd_k_particion_parcial`. Era INCORRECTA: esa función evalúa la
        # partición parcial dejando a los futuros aún no asignados con mecanismo
        # vacío (su pérdida máxima), por lo que es una cota *superior* del
        # subárbol, no inferior. Podar con `cota_superior >= mejor` recortaba
        # ramas que sí contenían el óptimo → el B&B sub-reportaba (devolvía
        # pérdidas mayores que el mínimo real). Sin una cota inferior admisible
        # para este objetivo, se enumera de forma exhaustiva: para 2N<=10 el
        # espacio es pequeño (la marginalización de n-cubos está cacheada) y el
        # resultado es exacto; para 2N>10 el `timeout` acota la búsqueda.

        siguiente = deltas[0]
        resto     = deltas[1:]
 
        for g_idx in range(k):
            if len(grupos[g_idx]) == 0:
                primer_vacio = next(i for i, g in enumerate(grupos) if len(g) == 0)
                if g_idx != primer_vacio:
                    continue
 
            grupos[g_idx].append(siguiente)
            self._bnb_explorar(grupos, resto, k)
            grupos[g_idx].pop()
    
    def _kpartir(self, grupos):
        """Construye el System k-partido: cada n-cubo futuro se marginaliza
        conservando solo las dimensiones presentes (mecanismo) de su propio grupo,
        forzando la independencia entre grupos."""
        mapa_efecto_a_mecanismo = {}

        for grupo in grupos:
            efectos = []
            mecanismos = []

            for tiempo, idx in grupo:
                if tiempo == EFFECT:
                    efectos.append(int(idx))
                else:
                    mecanismos.append(int(idx))

            mecanismos = np.array(sorted(mecanismos), dtype=np.int8)

            for efecto in efectos:
                mapa_efecto_a_mecanismo[int(efecto)] = mecanismos

        nuevo = System.__new__(System)
        nuevo.estado_inicial = self.sia_subsistema.estado_inicial
        nuevo.memo = {}

        nuevo.ncubos = tuple(
            cube.marginalizar(
                np.setdiff1d(
                    cube.dims,
                    mapa_efecto_a_mecanismo.get(
                        int(cube.indice),
                        np.array([], dtype=np.int8),
                    ),
                )
            )
            for cube in self.sia_subsistema.ncubos
        )

        return nuevo
 
    def _emd_k_particion_parcial(
        self, grupos_no_vacios: list[list]
    ) -> tuple[float, np.ndarray | None]:
        """Cota inferior parcial para el B&B: pérdida de los grupos ya formados
        (ignora los vértices sin asignar). Devuelve 0.0 si no es evaluable."""

        try:
            lb, _ = self._emd_k_particion(grupos_no_vacios)
            return lb, None
        except Exception:
            return 0.0, None

    def _emd_k_particion(
        self,
        grupos: list[list],
    ) -> tuple[float, np.ndarray]:
        """Pérdida de una k-partición: reconstruye el sistema con `_kpartir`, toma su
        distribución marginal y la compara con la del subsistema vía EMD-Efecto."""
 
        grupos_no_vacios = [list(g) for g in grupos if g]

        if not grupos_no_vacios:
            return INFTY_POS, np.array([])

        sistema_particionado = self._kpartir(grupos_no_vacios)
        dist_reconstruida = sistema_particionado.distribucion_marginal()

        if len(dist_reconstruida) != len(self.sia_dists_marginales):
            raise ValueError(
                f"Distribuciones incompatibles: "
                f"partición={len(dist_reconstruida)}, "
                f"subsistema={len(self.sia_dists_marginales)}"
            )

        perdida = emd_efecto(dist_reconstruida, self.sia_dists_marginales)

        return perdida, dist_reconstruida
 
    def _extraer_indices(
        self, grupo: list[tuple]
    ) -> tuple[list[int], list[int]]:
        """Separa un grupo en sus índices de alcance (futuro, EFFECT) y de mecanismo
        (presente, ACTUAL), ambos ordenados."""
  
        idxs_alcance    = sorted(idx for t, idx in grupo if t == EFFECT)
        dims_mecanismo  = sorted(idx for t, idx in grupo if t == ACTUAL)
        return idxs_alcance, dims_mecanismo
 
    def _clave_k(self, grupos: list[list]) -> tuple[frozenset, ...]:
        """Clave canónica (invariante al orden de grupos) de una k-partición, para memoización."""
        return tuple(sorted(frozenset(g) for g in grupos if g))
 
    def _fmt_k_particion(self, grupos: list[list]) -> str:
        """Formatea la k-partición como texto (futuros en mayúsculas arriba, presentes
        en minúsculas abajo, un bloque por grupo). Para k=2 usa `fmt_biparticion_q`."""
        if len(grupos) == 2:
            return fmt_biparticion_q(grupos[0], grupos[1])

        partes_superiores = []
        partes_inferiores = []

        for grupo in grupos:
            futuro = []
            presente = []

            for tiempo, idx in grupo:
                if tiempo == EFFECT:
                    futuro.append(ABECEDARY[int(idx)])
                else:
                    presente.append(ABECEDARY[int(idx)].lower())

            arriba = ",".join(sorted(futuro)) if futuro else "∅"
            abajo = ",".join(sorted(presente)) if presente else "∅"

            ancho = max(len(arriba), len(abajo), 1) + 2

            partes_superiores.append(f"|{arriba:^{ancho}}|")
            partes_inferiores.append(f"|{abajo:^{ancho}}|")

        linea_superior = "".join(partes_superiores)
        linea_inferior = "".join(partes_inferiores)
        divisor = "-" * len(linea_superior)

        return f"{linea_superior}\n{divisor}\n{linea_inferior}"

    def funcion_submodular(
        self,
        deltas: Union[tuple, list[tuple]],
        omegas: list[Union[tuple, list[tuple]]],
    ) -> tuple[float, float, np.ndarray]:
        """Función submodular de QNodes: devuelve (EMD de la unión omega∪delta, EMD del
        delta solo, distribución marginal del delta). Memoiza la EMD del delta."""

        vector_delta_marginal = None
        self.clave_submodular = [], []
 
        clave_delta_actual, clave_delta_efecto = self.definir_clave(deltas)
        clave_delta = tuple(clave_delta_actual), tuple(clave_delta_efecto)
 
        idxs_alcance_delta  = self.clave_submodular[EFFECT]
        dims_mecanismo_delta = self.clave_submodular[ACTUAL]
 
        if clave_delta not in self.memoria_delta:
            particion_delta = self.sia_subsistema.bipartir(
                np.array(idxs_alcance_delta,  dtype=np.int8),
                np.array(dims_mecanismo_delta, dtype=np.int8),
            )
            vector_delta_marginal = particion_delta.distribucion_marginal()
            emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)
            self.memoria_delta[clave_delta] = emd_delta, vector_delta_marginal
        else:
            emd_delta, vector_delta_marginal = self.memoria_delta[clave_delta]
 
        for omega in omegas:
            self.definir_clave(omega)
 
        idxs_alcance_union   = self.clave_submodular[EFFECT]
        dims_mecanismo_union = self.clave_submodular[ACTUAL]
 
        particion_union = self.sia_subsistema.bipartir(
            np.array(idxs_alcance_union,   dtype=np.int8),
            np.array(dims_mecanismo_union, dtype=np.int8),
        )
        vector_union_marginal = particion_union.distribucion_marginal()
        emd_union = emd_efecto(vector_union_marginal, self.sia_dists_marginales)
 
        return emd_union, emd_delta, vector_delta_marginal
 
    def definir_clave(
        self,
        conjunto: Union[tuple[int, int], list[tuple[int, int]], tuple[tuple[int, int], ...]],
    ):
        """Acumula en `clave_submodular` los índices de `conjunto` separados por tiempo
        (ACTUAL/EFFECT), aceptando un nodo simple (tiempo, idx) o una colección de nodos."""
        def es_nodo_simple(x):
            return (
                isinstance(x, tuple)
                and len(x) == 2
                and isinstance(x[0], (int, np.integer))
                and isinstance(x[1], (int, np.integer))
            )

        if es_nodo_simple(conjunto):
            tiempo, indice = conjunto
            self.clave_submodular[int(tiempo)].append(int(indice))
        else:
            for tiempo, indice in conjunto:
                self.clave_submodular[int(tiempo)].append(int(indice))

        self.clave_submodular[ACTUAL].sort()
        self.clave_submodular[EFFECT].sort()

        return self.clave_submodular
 
    def nodes_complement(self, nodes: list[tuple[int, int]]) -> list:
        """Devuelve los vértices del subsistema que no están en `nodes` (complemento)."""
        return list(set(self.vertices) - set(nodes))

