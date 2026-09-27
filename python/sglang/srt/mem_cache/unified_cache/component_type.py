"""The per-attention component identity; a leaf module importable from anywhere."""

from enum import Enum


class ComponentType(int, Enum):
    """Integer enum so that per-node list/tuple storage can be indexed directly."""

    FULL = 0
    SWA = 1
    MAMBA = 2
    C128 = 3
    # QSA compressed-K pages owned by the radix nodes (Qwen4-Exp stage 2).
    QSA_CP = 4

    def __str__(self) -> str:  # keep human-readable logging
        return self.name.lower()

    @property
    def is_full(self) -> bool:
        return self == ComponentType.FULL

    @property
    def is_swa(self) -> bool:
        return self == ComponentType.SWA

    @property
    def is_mamba(self) -> bool:
        return self == ComponentType.MAMBA

    @property
    def is_qsa_cp(self) -> bool:
        return self == ComponentType.QSA_CP


BASE_COMPONENT_TYPE = ComponentType.FULL
