# CAARE — Classiq Auto-Ansatz & Resource Explorer

A website that takes a combinatorial optimization problem and:
- Generates/loads the instance
- Solves it with a **classical baseline**
- Solves it with **QAOA on a simulator** (quantum side)
- Reports **resource estimates** (depth, 2-qubit gate count, width), and compares outcomes

> Roadmap: integrate **Classiq SDK** to synthesize and optimize circuits from a high-level problem spec (stubbed in `quantum/classiq_integration.py`).

## Quickstart

### 0) Prereqs
- Python 3.10–3.12 recommended
- A working C compiler is helpful for SciPy on some systems

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the app
```bash
streamlit run app.py
```

### 4) What works today
- Generate a random graph instance
- **Classical baseline** (brute-force up to ~18 nodes, greedy otherwise)
- **QAOA simulator** (Qiskit Aer) with p-layer choice and basic optimizer
- **Resource estimates** via transpilation (depth, 2-qubit gate count)

### 5) Coming next
- Classiq modeling → **auto-ansatz** generation and resource projection
- Additional problems (Knapsack/Portfolio as QUBO), kernels & explanations
- Noise-aware transpilation and error bars
