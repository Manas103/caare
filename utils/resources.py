from qiskit import transpile

def estimate_resources_qiskit(circuit, basis_gates=None, optimization_level=2):
    tq = transpile(circuit, basis_gates=basis_gates, optimization_level=optimization_level)
    depth = tq.depth()
    counts = tq.count_ops()
    twoq = 0
    for gate, c in counts.items():
        if gate.lower() in ["cx", "cz", "rzz", "swap", "csx"]:
            twoq += c
    return {
        "num_qubits": tq.num_qubits,
        "depth": depth,
        "two_qubit_gates": int(twoq),
        "op_counts": {k: int(v) for k, v in counts.items()}
    }
