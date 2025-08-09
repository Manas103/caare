import networkx as nx

def _maybe_call(x):
    return x() if callable(x) else x

def _find_qiskit_circuit(obj):
    """Return a Qiskit QuantumCircuit if present (possibly nested), else None."""
    try:
        from qiskit.circuit import QuantumCircuit as QC
    except Exception:
        QC = None
    if QC is not None and isinstance(obj, QC):
        return obj
    if hasattr(obj, "num_qubits") and (hasattr(obj, "qasm") or hasattr(obj, "count_ops")):
        return obj
    it = obj.values() if isinstance(obj, dict) else (obj if isinstance(obj, (list, tuple, set)) else [])
    for item in it:
        hit = _find_qiskit_circuit(item)
        if hit is not None:
            return hit
    return None

def _unwrap(obj):
    """Return a representative inner object from tuples/lists/dicts."""
    if isinstance(obj, (list, tuple)):
        for item in obj:
            hit = _unwrap(item)
            if any(hasattr(hit, a) for a in ("num_qubits", "count_ops", "depth", "to_qiskit", "qasm")):
                return hit
        return obj[0]
    if isinstance(obj, dict) and obj:
        for v in obj.values():
            hit = _unwrap(v)
            if any(hasattr(hit, a) for a in ("num_qubits", "count_ops", "depth", "to_qiskit", "qasm")):
                return hit
        return next(iter(obj.values()))
    return obj

def _coerce_to_qiskit(obj):
    """Try very hard to obtain a Qiskit QuantumCircuit; else return None."""
    obj = _unwrap(obj)
    hit = _find_qiskit_circuit(obj)
    if hit is not None:
        return hit
    for m in ("to_qiskit", "qiskit", "as_qiskit", "qiskit_circuit"):
        if hasattr(obj, m):
            try:
                qc = getattr(obj, m)()
                qc = _find_qiskit_circuit(qc) or _unwrap(qc)
                if qc is not None:
                    return qc
            except Exception:
                pass
    return None

def _extract_resources_generic(circ):
    """
    Compute (depth, num_qubits, op_counts) for either Qiskit or generic objects.
    Returns (depth, num_qubits, dict(op->count)).
    """
    circ = _unwrap(circ)

    try:
        from qiskit import transpile
        from qiskit.circuit import QuantumCircuit as QC
        if isinstance(circ, QC):
            tq = transpile(circ, optimization_level=1)
            return tq.depth(), tq.num_qubits, {k: int(v) for k, v in tq.count_ops().items()}
    except Exception:
        pass

    depth = _maybe_call(getattr(circ, "depth", None))
    num_qubits = _maybe_call(getattr(circ, "num_qubits", None))
    if num_qubits in (None, 0, False):
        qattr = getattr(circ, "qubits", None)
        if isinstance(qattr, (list, tuple)):
            num_qubits = len(qattr)
        elif isinstance(qattr, int):
            num_qubits = qattr
        else:
            qregs = getattr(circ, "qregs", None)
            if qregs:
                try:
                    num_qubits = sum(getattr(reg, "size", 0) for reg in qregs)
                except Exception:
                    pass

    counts_attr = getattr(circ, "count_ops", None)
    if callable(counts_attr):
        try:
            counts = counts_attr()
        except Exception:
            counts = {}
    else:
        counts = getattr(circ, "op_counts", {}) or {}

    norm_counts = {}
    if isinstance(counts, dict):
        for k, v in counts.items():
            try:
                norm_counts[k] = int(v)
            except Exception:
                pass
    return depth, num_qubits, norm_counts


def build_pyomo_maxcut_model(G: nx.Graph):
    import pyomo.environ as pyo
    m = pyo.ConcreteModel()
    nodes = list(G.nodes())
    m.x = pyo.Var(nodes, domain=pyo.Binary)

    def obj_rule(mm):
        return sum(mm.x[i] + mm.x[j] - 2 * mm.x[i] * mm.x[j] for (i, j) in G.edges())

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    return m


def build_pyomo_knapsack_model(values, weights, capacity):
    import pyomo.environ as pyo
    n = len(values)
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, domain=pyo.Binary)
    m.value = pyo.Param(m.I, initialize=lambda _, i: int(values[i]), within=pyo.NonNegativeIntegers, mutable=True)
    m.weight = pyo.Param(m.I, initialize=lambda _, i: int(weights[i]), within=pyo.NonNegativeIntegers, mutable=True)
    m.capacity = pyo.Param(initialize=int(capacity), within=pyo.NonNegativeIntegers, mutable=True)
    m.obj = pyo.Objective(expr=sum(m.value[i] * m.x[i] for i in m.I), sense=pyo.maximize)
    m.cap = pyo.Constraint(expr=sum(m.weight[i] * m.x[i] for i in m.I) <= m.capacity)
    return m


def _synthesize_and_resources(
    qmod,
    num_qubits_fallback=None,
    *,
    do_execute: bool = False,
    backend: str | None = None,
):
    """
    Synthesize with Classiq and return resource metrics.
    Execution is optional (off by default to avoid backend errors).

    Returns dict with:
      ok, executed, resources{ num_qubits, depth, two_qubit_gates|None, op_counts },
      [energy, iterations, time_sec], qiskit_circuit (if available),
      [execution_error] when do_execute=True and execution fails.
    """
    from classiq import synthesize
    qprog = synthesize(qmod)

    raw_circ = getattr(qprog, "transpiled_circuit", None) or getattr(qprog, "circuit", None) or qprog
    qc = _coerce_to_qiskit(raw_circ)
    depth, num_qubits, op_counts = _extract_resources_generic(qc or raw_circ)

    if (num_qubits is None or (isinstance(num_qubits, (int, float)) and int(num_qubits) == 0)) and num_qubits_fallback is not None:
        num_qubits = int(num_qubits_fallback)

    twoq = None
    if isinstance(op_counts, dict) and len(op_counts) > 0:
        twoq = 0
        for gate, c in op_counts.items():
            if str(gate).lower() in ["cx", "cz", "rzz", "swap", "csx", "ecr", "zz"]:
                twoq += int(c)

    resources = {
        "num_qubits": int(num_qubits) if isinstance(num_qubits, (int, float)) else None,
        "depth": int(depth) if isinstance(depth, (int, float)) else None,
        "two_qubit_gates": twoq,  # None → N/A
        "op_counts": op_counts,
    }

    if not do_execute:
        return {
            "ok": True,
            "executed": False,
            "resources": resources,
            "qiskit_circuit": qc or _unwrap(raw_circ),
        }

    try:
        from classiq import execute
        kwargs = {}
        if backend:
            kwargs["backend"] = backend
        res = execute(qprog, **kwargs).result()
        if res and len(res) > 0 and hasattr(res[0], "value"):
            r0 = res[0].value
            energy = getattr(r0, "energy", None)
            time_sec = getattr(r0, "time", None)
            inter = getattr(r0, "intermediate_results", None)
            iterations = None
            if inter:
                try:
                    iterations = int(inter[-1].iteration_number)
                except Exception:
                    iterations = len(inter)
            return {
                "ok": True,
                "executed": True,
                "resources": resources,
                "energy": float(energy) if energy is not None else None,
                "iterations": iterations,
                "time_sec": float(time_sec) if time_sec is not None else None,
                "qiskit_circuit": qc or _unwrap(raw_circ),
            }
        return {
            "ok": True,
            "executed": True,
            "resources": resources,
            "qiskit_circuit": qc or _unwrap(raw_circ),
        }
    except Exception as e:
        return {
            "ok": True,
            "executed": False,
            "execution_error": str(e),
            "resources": resources,
            "qiskit_circuit": qc or _unwrap(raw_circ),
        }


def classiq_qaoa_maxcut(
    G: nx.Graph,
    p: int = 2,
    shots: int = 1024,
    max_iter: int = 120,
    *,
    do_execute: bool = False,
    backend: str | None = None,
):
    """
    Build a Pyomo Max-Cut model, synthesize with Classiq QAOA, and return resources.
    Execution is disabled by default to avoid backend errors.
    """
    try:
        from classiq import construct_combinatorial_optimization_model
        from classiq.applications.combinatorial_optimization import QAOAConfig, OptimizerConfig
        import classiq
        # best-effort auth (no-op if already authenticated)
        if hasattr(classiq, "authenticate"):
            try:
                classiq.authenticate(overwrite=False)
            except Exception:
                pass
    except Exception as e:
        return {"ok": False, "error": f"Classiq SDK not available: {e}"}

    qmod = construct_combinatorial_optimization_model(
        pyo_model=build_pyomo_maxcut_model(G),
        qaoa_config=QAOAConfig(num_layers=int(p)),
        optimizer_config=OptimizerConfig(max_iteration=int(max_iter)),
    )
    return _synthesize_and_resources(
        qmod,
        num_qubits_fallback=G.number_of_nodes(),
        do_execute=do_execute,
        backend=backend,
    )


def classiq_qaoa_knapsack(
    values,
    weights,
    capacity,
    p: int = 2,
    shots: int = 1024,
    max_iter: int = 120,
    *,
    do_execute: bool = False,
    backend: str | None = None,
):
    """
    Build a Pyomo 0/1 Knapsack model, synthesize with Classiq QAOA, and return resources.
    Execution is disabled by default to avoid backend errors.
    """
    try:
        from classiq import construct_combinatorial_optimization_model
        from classiq.applications.combinatorial_optimization import QAOAConfig, OptimizerConfig
        import classiq
        if hasattr(classiq, "authenticate"):
            try:
                classiq.authenticate(overwrite=False)
            except Exception:
                pass
    except Exception as e:
        return {"ok": False, "error": f"Classiq SDK not available: {e}"}

    qmod = construct_combinatorial_optimization_model(
        pyo_model=build_pyomo_knapsack_model(values, weights, capacity),
        qaoa_config=QAOAConfig(num_layers=int(p)),
        optimizer_config=OptimizerConfig(max_iteration=int(max_iter)),
    )
    return _synthesize_and_resources(
        qmod,
        num_qubits_fallback=len(values),
        do_execute=do_execute,
        backend=backend,
    )
