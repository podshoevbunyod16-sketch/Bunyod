"""State-graph DAG orchestrator supporting sequential & parallel nodes, retries, cancel & recovery."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class GraphNode:
    id: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    depends_on: List[str] = field(default_factory=list)
    retries: int = 0
    timeout_sec: float = 60.0


@dataclass
class GraphExecutionResult:
    status: str  # completed | failed | cancelled
    state: Dict[str, Any]
    node_outputs: Dict[str, Dict[str, Any]]
    execution_order: List[str]
    errors: Dict[str, str]
    duration_sec: float


class StateGraph:
    """
    Specification Section 2 & 5:
    Deterministic DAG state orchestrator with cycle detection, topological waves (parallel execution
    for independent nodes), retry handling, cancellation, and checkpointable state dictionary.
    """

    def __init__(self, name: str = "workflow") -> None:
        self.name = name
        self._nodes: Dict[str, GraphNode] = {}

    def add_node(
        self,
        node_id: str,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        depends_on: Optional[List[str]] = None,
        retries: int = 0,
        timeout_sec: float = 60.0,
    ) -> "StateGraph":
        self._nodes[node_id] = GraphNode(
            id=node_id,
            fn=fn,
            depends_on=list(depends_on or []),
            retries=retries,
            timeout_sec=timeout_sec,
        )
        return self

    def validate(self) -> List[List[str]]:
        """Validate dependencies & return parallelizable topological waves; raises ValueError on cycle."""
        for nid, node in self._nodes.items():
            for dep in node.depends_on:
                if dep not in self._nodes:
                    raise ValueError(f"Node '{nid}' depends on unknown node '{dep}'.")

        remaining: Set[str] = set(self._nodes.keys())
        completed: Set[str] = set()
        waves: List[List[str]] = []

        while remaining:
            ready = sorted(
                nid for nid in remaining if all(dep in completed for dep in self._nodes[nid].depends_on)
            )
            if not ready:
                raise ValueError(f"Cycle detected in StateGraph '{self.name}' among nodes: {sorted(remaining)}")
            waves.append(ready)
            for nid in ready:
                remaining.remove(nid)
                completed.add(nid)
        return waves

    def run(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[threading.Event] = None,
        max_parallel: int = 4,
    ) -> GraphExecutionResult:
        waves = self.validate()
        state: Dict[str, Any] = dict(initial_state or {})
        outputs: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        errors: Dict[str, str] = {}
        state_lock = threading.Lock()
        t0 = time.monotonic()

        def _exec_node(node: GraphNode) -> Dict[str, Any]:
            attempt = 0
            while True:
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError(f"Graph '{self.name}' cancelled before node '{node.id}'.")
                try:
                    with state_lock:
                        snapshot = dict(state)
                    return node.fn(snapshot) or {}
                except InterruptedError:
                    raise
                except Exception as exc:
                    if attempt < node.retries:
                        attempt += 1
                        time.sleep(0.05)
                        continue
                    raise exc

        for wave in waves:
            if cancel_event and cancel_event.is_set():
                return GraphExecutionResult(
                    status="cancelled",
                    state=state,
                    node_outputs=outputs,
                    execution_order=order,
                    errors=errors,
                    duration_sec=round(time.monotonic() - t0, 4),
                )
            if len(wave) == 1:
                nid = wave[0]
                try:
                    out = _exec_node(self._nodes[nid])
                    outputs[nid] = out
                    state.update(out)
                    order.append(nid)
                except InterruptedError as exc:
                    errors[nid] = str(exc)
                    return GraphExecutionResult(
                        status="cancelled",
                        state=state,
                        node_outputs=outputs,
                        execution_order=order,
                        errors=errors,
                        duration_sec=round(time.monotonic() - t0, 4),
                    )
                except Exception as exc:
                    errors[nid] = f"{type(exc).__name__}: {exc}"
                    return GraphExecutionResult(
                        status="failed",
                        state=state,
                        node_outputs=outputs,
                        execution_order=order,
                        errors=errors,
                        duration_sec=round(time.monotonic() - t0, 4),
                    )
            else:
                with ThreadPoolExecutor(max_workers=min(max_parallel, len(wave))) as pool:
                    fut_map = {pool.submit(_exec_node, self._nodes[nid]): nid for nid in wave}
                    for fut in as_completed(fut_map):
                        nid = fut_map[fut]
                        try:
                            out = fut.result(timeout=self._nodes[nid].timeout_sec)
                            with state_lock:
                                outputs[nid] = out
                                state.update(out)
                                order.append(nid)
                        except Exception as exc:
                            errors[nid] = f"{type(exc).__name__}: {exc}"
                if errors:
                    return GraphExecutionResult(
                        status="failed",
                        state=state,
                        node_outputs=outputs,
                        execution_order=order,
                        errors=errors,
                        duration_sec=round(time.monotonic() - t0, 4),
                    )

        return GraphExecutionResult(
            status="completed",
            state=state,
            node_outputs=outputs,
            execution_order=order,
            errors=errors,
            duration_sec=round(time.monotonic() - t0, 4),
        )
