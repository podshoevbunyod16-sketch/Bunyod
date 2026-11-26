"""RASM-CIM v0.2 — integrated prototype. v0.1 preserved in rasm_cim/."""
from .rns2 import PackedRNS, BinaryBaseline
from .cam2 import CAM2
from .ssm2 import SelectiveSSM, LinearSSMv1, TanhRNNBaseline
from .resonator2 import Resonator2
from .logic2 import LogicMemory2
from .program2 import ProgramMemory2, execute_program, parse_program_string
from .system2 import RASMCIMv2

__version__ = "0.2.0"
