"""Configuración de pytest: hace importable el paquete `src` desde la raíz
del subproyecto Method2 (un nivel por encima de este archivo)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
