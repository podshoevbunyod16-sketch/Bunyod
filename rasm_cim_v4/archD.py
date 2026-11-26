"""
Arch-D: Local Rule Graph Machine (LRGM) — independent of RNS/CAM/HDC.

Principle: knowledge = graph; computation = local message passing (BFS),
rule rewriting, and BFS program synthesis. No RNS residues, no CAM,
no hypervectors, no matrix multiplication in reasoning path.
Learning = edge insertion + BFS synthesis from examples.
Sequences = activation propagation on graph (local updates).
"""
from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set
from rasm_cim_v4.graph4 import TruthGraph
from rasm_cim_v4.program4 import synthesize_split
from rasm_cim_v3.program3 import execute


class ArchD:
    def __init__(self, seed: int = 0):
        self.graph = TruthGraph(functional_relations={0,1,2})
        self.tools = {"add": lambda a,b: int(a)+int(b), "mul": lambda a,b: int(a)*int(b)}

    # --- knowledge ---
    def learn_fact(self, s, r, o): self.graph.assert_true(s, r, o)
    def learn_false(self, s, r, o): self.graph.assert_false(s, r, o)
    def retract(self, s, r, o): self.graph.retract(s, r, o)
    def query(self, s, r, o=None): return self.graph.query(s, r, o)
    def compose(self, s, r1, r2): return self.graph.compose(s, r1, r2)
    def compose_value(self, s, r1, r2): return self.graph.compose_value(s, r1, r2)
    def reachable(self, s, depth=3): return self.graph.reachable(s, depth)

    # --- sequences via local activation propagation (no matrices) ---
    def propagate(self, start: int, steps: int = 3) -> Set[int]:
        """Local propagation: activate neighbors step by step (BFS)."""
        return self.graph.reachable(start, max_depth=steps)

    # --- programs via BFS synthesis (graph search, no HV) ---
    def synthesize(self, train, val, test, max_depth=3):
        return synthesize_split(train, val, test, max_depth=max_depth)

    def run_program(self, program, x): return execute(program, x)

    # --- tools via graph registry (no CAM) ---
    def call_tool(self, name: str, *args):
        if name not in self.tools: raise ValueError(f"Unknown tool {name}")
        return self.tools[name](*args)

    def save(self):
        return {"history": self.graph.save_history()}

    def load(self, state):
        # replay history (deterministic rebuild)
        self.graph = TruthGraph(functional_relations={0,1,2})
        for h in state["history"]:
            if h["op"]=="assert_true": self.graph.assert_true(h["s"],h["r"],h["o"])
            elif h["op"]=="assert_false": self.graph.assert_false(h["s"],h["r"],h["o"])
            elif h["op"]=="retract": self.graph.retract(h["s"],h["r"],h["o"])
