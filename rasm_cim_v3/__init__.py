"""RASM-CIM v0.3 platform. v0.1 in rasm_cim/, v0.2 in rasm_cim_v2/ (preserved)."""
from .memory3 import RNSv3, MultiBankRNS, FixedCAM, GraphMemory
from .ssm3 import TanhRNNBPTT, make_parity
from .resonator3 import Resonator3
from .program3 import synthesize, execute
from .arch import ArchA, ArchB, ArchC

__version__ = "0.3.0"
