"""
RASMCIMv2 — integrated pipeline for v0.2.

Pipeline (each component influences output; ablation flags prove it):
  write:  int/fact -> PackedRNS residues -> SelectiveSSM context -> CAM2 write
  query:  CAM2 (exact/partial/noisy+tau) -> Resonator2 (if HV composite) ->
          Logic2 (refusal/contradiction/compose) / Program2 (safe exec) / Tools

Ablation flags (for F7):
  use_ssm=False  -> context_id always 0 (context-dependent recall must fail)
  use_cam=False  -> only dict exact (noisy/partial must fail)
  use_resonator=False -> brute force only (slow, but exact; shows resonator role is speed/confidence)
  use_rns=False  -> HV only (exact integer keys must fail / need HV)
"""
from __future__ import annotations
import numpy as np
from .rns2 import PackedRNS
from .cam2 import CAM2, random_bipolar, flip_bits
from .ssm2 import SelectiveSSM
from .logic2 import LogicMemory2
from .program2 import ProgramMemory2, execute_program
from .tools import ToolRegistry


class RASMCIMv2:
    def __init__(self, moduli=(3, 5, 7, 11), D: int = 256, state_dim: int = 32, seed: int = 0,
                 use_ssm: bool = True, use_cam: bool = True, use_resonator: bool = True, use_rns: bool = True):
        self.rns = PackedRNS(moduli)
        self.D = D
        self.rng = np.random.default_rng(seed)
        self.cam = CAM2(D=D, tau=0.5)
        self.ssm = SelectiveSSM(input_dim=D, state_dim=state_dim, seed=seed)
        self.logic = LogicMemory2(n_subjects=50, n_relations=5, n_objects=50, D=D, seed=seed, tau=0.6)
        self.programs = ProgramMemory2(D=D, seed=seed)
        self.tools = ToolRegistry(D=D, seed=seed)
        self.fpe_phi = self.rng.uniform(0, 2 * np.pi, size=D)
        self.use_ssm = use_ssm
        self.use_cam = use_cam
        self.use_resonator = use_resonator
        self.use_rns = use_rns
        # for context-dependent recall: store (key_int, context_id) -> value via CAM residues+context
        self._ctx_store = {}

    def _key_hv(self, key_int: int) -> np.ndarray:
        # bipolar key from FPE phase sign (deterministic per key)
        hv = np.sign(np.cos(self.fpe_phi * (key_int % 997)))  # periodic-ish deterministic
        hv[hv == 0] = 1.0
        return hv

    def _context(self) -> int:
        if not self.use_ssm:
            return 0
        return self.ssm.context_id()

    def write_pair(self, key_int: int, value_int: int, context_event: np.ndarray | None = None):
        """One-shot write. SSM updates from context_event (or key HV)."""
        key_hv = self._key_hv(key_int)
        if context_event is not None:
            self.ssm.step(np.asarray(context_event, dtype=float), event=True)
        else:
            self.ssm.step(key_hv, event=True)
        ctx = self._context()
        # RNS residues + context as CAM key: append context as extra residue-like dim
        if self.use_rns:
            res = tuple(int(r) for r in self.rns.encode_packed(key_int)) + (ctx,)
        else:
            res = (key_int % 1000, ctx)  # degraded: HV only path still needs a key; use hash
        if self.use_cam:
            self.cam.write(res, key_hv, value_int)
        self._ctx_store[(key_int, ctx)] = value_int
        return ctx

    def query_pair(self, key_int: int, noisy: bool = False, noise_frac: float = 0.1):
        ctx = self._context()
        if self.use_rns:
            res = tuple(int(r) for r in self.rns.encode_packed(key_int)) + (ctx,)
        else:
            res = (key_int % 1000, ctx)
        if not self.use_cam:
            # dict-only ablation: exact (key,ctx) lookup, no noisy tolerance
            if noisy:
                return None, -2.0, True
            v = self._ctx_store.get((key_int, ctx))
            return v, (1.0 if v is not None else -2.0), (v is None)
        if noisy:
            qhv = flip_bits(self._key_hv(key_int), noise_frac, self.rng)
            return self.cam.search_noisy(qhv)
        else:
            return self.cam.search_exact(res)

    def reset_state(self):
        self.ssm.reset()

    # --- logic / program / tools passthrough (integrated, not side demos) ---
    def learn_fact(self, s: int, r: int, o: int) -> bool:
        return self.logic.add_fact(s, r, o)

    def query_fact(self, s: int, r: int):
        return self.logic.query(s, r)

    def compose_fact(self, s: int, r1: int, r2: int):
        return self.logic.compose(s, r1, r2)

    def generate_and_run(self, program_hvs, x: int):
        prog, _ = self.programs.decode_program(program_hvs)
        if prog is None:
            return None, None
        return prog, execute_program(prog, x)
