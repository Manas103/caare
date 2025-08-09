import pandas as pd
import numpy as np
import networkx as nx

from classical.maxcut_bruteforce import solve_maxcut_classical
from quantum.qaoa_maxcut import qaoa_solve_maxcut, qaoa_build_param_circuit
from utils.resources import estimate_resources_qiskit

def sweep_qaoa(G: nx.Graph, p_values, shots=1024, max_iter=80, trials=3, seed=42):
    rng = np.random.default_rng(int(seed))
    rows = []
    classical = solve_maxcut_classical(G, time_budget_s=8.0)
    best_classical = classical["best_value"]
    for p in p_values:
        circ = qaoa_build_param_circuit(G, p, betas=[0.1]*p, gammas=[0.1]*p)
        res = estimate_resources_qiskit(circ)
        for t in range(int(trials)):
            s = int(rng.integers(0, 2**31 - 1))
            qres = qaoa_solve_maxcut(G, p=p, shots=shots, max_iter=max_iter, seed=s)
            rows.append({
                "p": int(p),
                "trial": int(t),
                "qaoa_best_value": int(qres["best_value"]),
                "approx_ratio_vs_edges": float(qres["approx_ratio"]),
                "approx_ratio_vs_classical": float(qres["best_value"]/best_classical) if best_classical > 0 else 1.0,
                "depth": int(res["depth"]),
                "two_qubit_gates": int(res["two_qubit_gates"]),
                "num_qubits": int(res["num_qubits"]),
                "iterations": int(qres["iterations"]) if qres["iterations"] is not None else None,
            })
    df = pd.DataFrame(rows)
    return df, classical
