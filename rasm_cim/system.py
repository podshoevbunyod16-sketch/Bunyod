"""
Integrated RASM-CIM system (prototype v0.1).

Pipeline: integer -> RNS residues -> CAM write -> SSM state update ->
CAM/resonator query -> output.

This is a research prototype, not a trained LLM. Learning = one-shot CAM write.
No gradient training is performed.
"""
from __future__ import annotations
import numpy as np
from .rns import RNS
from .cam import AssociativeMemory, random_bipolar, flip_bits
from .state import DiagonalSSM
from .fpe import IntegerFPE


class RASMCIM:
    def __init__(self, moduli=(3, 5, 7, 11), D: int = 512, state_dim: int = 32, seed: int = 0):
        self.rns = RNS(moduli)
        self.D = D
        self.rng = np.random.default_rng(seed)
        self.cam = AssociativeMemory(D=D)
        # SSM input is the projected HV (state_dim,), so input_dim=state_dim
        self.ssm = DiagonalSSM(input_dim=state_dim, state_dim=state_dim, decay=0.9, seed=seed)
        self.fpe = IntegerFPE(D=D, seed=seed)
        # random projection for SSM input: HV -> state input (fixed, untrained)
        self.proj = self.rng.normal(0, 1.0 / np.sqrt(D), size=(state_dim, D))

    def write_pair(self, key_int: int, value_int: int):
        """One-shot write: key HV -> value, plus RNS residue index, plus SSM update."""
        key_hv = self.fpe.encode(key_int)
        # store bipolar version for CAM (sign of real part as bipolar proxy)
        # For prototype simplicity, store random bipolar keyed by key_int via FPE phase sign
        bipolar_key = np.sign(np.real(key_hv))
        bipolar_key[bipolar_key == 0] = 1.0
        self.cam.write_hv(bipolar_key, value_int)
        self.cam.write_residue(self.rns.encode(key_int), value_int)
        # SSM update with key HV (real+imag stacked? use real part)
        x = np.real(key_hv)
        self.ssm.step(self.proj @ x, event=True)

    def query_exact(self, key_int: int):
        return self.cam.search_dict(self.rns.encode(key_int))

    def query_noisy(self, key_int: int, noise_frac: float = 0.1):
        key_hv = self.fpe.encode(key_int)
        bipolar_key = np.sign(np.real(key_hv))
        bipolar_key[bipolar_key == 0] = 1.0
        noisy = flip_bits(bipolar_key, noise_frac, self.rng)
        val, sim = self.cam.search_cam_vectorized(noisy)
        return val, sim

    def reset_state(self):
        self.ssm.reset()
