"""
Program synthesis v0.3 via BFS search over ops (graph search, not HV brute force).

Fixes v0.2: MAX_STEPS 3 -> up to 5, synthesis from input-output examples
by BFS over program space (each node = program, edges = append op).
Execution via whitelist interpreter (no eval). Search is exact for small
depths; complexity O(B^depth) documented honestly.
"""
from __future__ import annotations
from collections import deque
from typing import List, Tuple, Optional

OPS_ARG = {"add","sub","mul","max","min"}
OPS_NOARG = {"neg","square","abs"}
OPS = ["add","sub","mul","max","min","neg","square","abs"]
ARG_VALUES = [-3,-2,-1,0,1,2,3,5]
ARG_MIN, ARG_MAX = -10, 10
VAL_MIN, VAL_MAX = -10**6, 10**6


class SafeError(ValueError):
    pass


def apply_op(op: str, arg, x: int) -> int:
    if op not in OPS: raise SafeError(f"bad op {op}")
    if not isinstance(x, int) or isinstance(x, bool): raise SafeError("x must be int")
    if op in OPS_ARG:
        if not isinstance(arg, int) or isinstance(arg, bool): raise SafeError("arg must be int")
        if not (ARG_MIN <= arg <= ARG_MAX): raise SafeError("arg OOB")
    if op=="add": y=x+arg
    elif op=="sub": y=x-arg
    elif op=="mul": y=x*arg
    elif op=="max": y=max(x,arg)
    elif op=="min": y=min(x,arg)
    elif op=="neg": y=-x
    elif op=="square": y=x*x
    elif op=="abs": y=abs(x)
    else: raise SafeError("unhandled")
    if not (VAL_MIN <= y <= VAL_MAX): raise SafeError("value OOB")
    return int(y)


def execute(program: List[Tuple[str, int|None]], x: int) -> int:
    if not isinstance(program, list) or not (1 <= len(program) <= 5):
        raise SafeError("program length 1..5")
    y = x
    for op, arg in program:
        y = apply_op(op, arg, y)
    return y


def synthesize(examples: List[Tuple[int,int]], max_depth: int = 3, max_nodes: int = 20000) -> Optional[list]:
    """
    BFS synthesis: find shortest program mapping all inputs to outputs.
    examples: [(x_in, y_out), ...]. Returns program or None.
    Ops expansion: arg-ops x ARG_VALUES + noarg-ops.
    """
    # BFS over programs; state = tuple of current values for all examples
    start_state = tuple(x for x,_ in examples)
    target = tuple(y for _,y in examples)
    if start_state == target:
        return []  # identity (empty program) — caller handles
    queue = deque()
    queue.append(([], start_state))
    visited = {start_state}
    nodes = 0
    while queue:
        prog, state = queue.popleft()
        if len(prog) >= max_depth: continue
        # expand
        candidates = []
        for op in OPS:
            if op in OPS_ARG:
                for a in ARG_VALUES:
                    candidates.append((op, a))
            else:
                candidates.append((op, None))
        for op, arg in candidates:
            nodes += 1
            if nodes > max_nodes:
                return None
            try:
                new_state = tuple(apply_op(op, arg, v) for v in state)
            except SafeError:
                continue
            new_prog = prog + [(op, arg)]
            if new_state == target:
                return new_prog
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_prog, new_state))
    return None


def program_str(program) -> str:
    return " -> ".join(f"{op}({a})" if a is not None else f"{op}()" for op,a in program) if program else "identity"
