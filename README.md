# CAARE — Classiq Auto-Ansatz & Resource Explorer

It's an interactive Streamlit app to **model, solve, and analyze** combinatorial optimization problems with both **classical** and **quantum (QAOA)** approaches, plus **Classiq SDK** for auto-ansatz synthesis and a **resource explorer**.

## What it does

- **Problems:** Max-Cut and **Knapsack (QUBO)**
- **Quantum path:** QAOA on Qiskit Aer (choose p, shots, optimizer iters)
- **Classiq integration:** Synthesize circuits from a Pyomo model (auto-ansatz), fetch **resource estimates** (depth, width, op counts)
- **Compare & summarize:** Automatic table and summary bullets
- **Resource explorer:** Transpile and count gates
- **Noise study:** Depolarizing model sweeps with **heatmaps**. Optional **reopt under noise** with progress bars
- **Benchmarking:** p-sweep with charts (quality vs. two-qubit gates)

---

### **0) Prerequisites**
- Python **3.10–3.12** recommended
- Build tools for SciPy/Qiskit may be required on Windows/macOS (MSVC / Xcode CLT)
- No quantum hardware account required

### **1) Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### **2) Install dependencies**
```bash
pip install -r requirements.txt
```

### **3) (Optional) Classiq SDK**
For auto-ansatz synthesis and Classiq resource estimates:
```bash
pip install classiq pyomo
```
Authenticate once:
```bash
# Option A: Let the app trigger login on first synthesis
# Option B: Authenticate manually
python -c "import classiq; classiq.authenticate()"
```
> If the SDK can’t return a Qiskit circuit for your version, the app will still display Classiq resource metrics (two-qubit gates may be N/A).

### **4) Run the app**
```bash
streamlit run app.py
```

---

## Using the App

1. **Choose a problem** (Max-Cut or Knapsack) in the sidebar and set instance parameters.  
2. **Click “Generate & Solve”** — the instance is stored in session state.  
3. Explore the tabs:  
   - **Solve:** Classical baseline, our QAOA, Classiq synthesis, auto-compare summary  
   - **Benchmark:** p-sweep plots  
   - **Noise:** Heatmaps over depolarizing probabilities; optional re-optimization  
   - **Hardware:** Post-layout metrics using fake IBM devices  

---

### 5) Coming next
- More problem templates (Max-k-Cut, MIS, Portfolio Optimization)
- VQE baseline + cost-landscape visualization
- Real-hardware submissions via providers
- One-click export of results/plots
