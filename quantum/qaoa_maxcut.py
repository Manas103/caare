import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit_aer import Aer
from qiskit import transpile
from scipy.optimize import minimize

def maxcut_cost_from_bitstring(G: nx.Graph, bitstring):
    value = 0
    for u, v in G.edges():
        if bitstring[u] != bitstring[v]:
            value += 1
    return value

def sample_expectation(G, counts):
    total = sum(counts.values())
    exp = 0.0
    for bitstr, c in counts.items():
        bits = list(map(int, bitstr[::-1]))
        exp += (c / total) * maxcut_cost_from_bitstring(G, bits)
    return exp

def qaoa_layer(circ, G, beta, gamma):
    for (u, v) in G.edges():
        circ.append(RZZGate(2 * gamma), [u, v])
    for q in range(G.number_of_nodes()):
        circ.rx(2 * beta, q)

def qaoa_build_param_circuit(G: nx.Graph, p: int, betas, gammas):
    n = G.number_of_nodes()
    circ = QuantumCircuit(n)
    for q in range(n):
        circ.h(q)
    for layer in range(p):
        qaoa_layer(circ, G, betas[layer], gammas[layer])
    circ.measure_all()
    return circ

def qaoa_solve_maxcut(G: nx.Graph, p=2, shots=1024, max_iter=80, seed=42):
    rng = np.random.default_rng(seed)
    betas = rng.uniform(0, np.pi, size=p)
    gammas = rng.uniform(0, np.pi, size=p)

    backend = Aer.get_backend("qasm_simulator")

    def objective(x):
        b = x[:p]
        g = x[p:]
        circ = qaoa_build_param_circuit(G, p, b, g)
        tq = transpile(circ, backend)
        result = backend.run(tq, shots=shots, seed_simulator=seed).result()
        counts = result.get_counts()
        return -sample_expectation(G, counts)

    x0 = np.concatenate([betas, gammas])

    res = minimize(
        objective, x0, method="COBYLA",
        options=dict(maxiter=int(max_iter), rhobeg=0.5)
    )

    iterations = getattr(res, "nit", None)
    if iterations is None:
        iterations = getattr(res, "nfev", None)

    b_opt = res.x[:p]
    g_opt = res.x[p:]

    circ = qaoa_build_param_circuit(G, p, b_opt, g_opt)
    tq = transpile(circ, backend)
    result = backend.run(tq, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()

    best_bitstring = max(counts, key=counts.get)
    bits = list(map(int, best_bitstring[::-1]))
    best_value = maxcut_cost_from_bitstring(G, bits)

    approx_ratio = best_value / G.number_of_edges() if G.number_of_edges() > 0 else 1.0

    return dict(
        best_value=best_value,
        best_bitstring=best_bitstring,
        approx_ratio=approx_ratio,
        iterations=iterations,
        fun=float(res.fun) if hasattr(res, "fun") else None,
        success=bool(getattr(res, "success", False)),
        message=str(getattr(res, "message", "")),
    )
