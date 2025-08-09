import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd

from classical.maxcut_bruteforce import solve_maxcut_classical
from classical.knapsack_dp import solve_knapsack_dp
from quantum.qaoa_maxcut import (
    qaoa_solve_maxcut,
    qaoa_build_param_circuit,
    sample_expectation,
)
from quantum.classiq_integration import (
    classiq_qaoa_maxcut,
    classiq_qaoa_knapsack,
)
from utils.resources import estimate_resources_qiskit
from utils.benchmark import sweep_qaoa


st.set_page_config(page_title="CAARE — Classiq Auto-Ansatz & Resource Explorer", layout="wide")

st.title("CAARE — Classiq Auto-Ansatz & Resource Explorer")
st.caption("Max-Cut • Knapsack (QUBO) • Classical vs QAOA • Resource estimates • Classiq auto-ansatz")

with st.sidebar:
    st.header("Problem")
    problem = st.selectbox("Choose problem", ["Max-Cut", "Knapsack (QUBO)"], index=0)

    st.header("Instance")
    seed = st.number_input("Random seed", value=42, step=1)

    if problem == "Max-Cut":
        n = st.slider("Nodes (n)", 4, 22, 10)
        p_edge = st.slider("Edge probability", 0.05, 0.9, 0.3)
    else:
        n = st.slider("Items (n)", 4, 25, 10)
        val_min, val_max = st.slider("Value range", 1, 50, (5, 20))
        wt_min, wt_max = st.slider("Weight range", 1, 50, (3, 15))
        cap_ratio = st.slider("Capacity ratio (vs sum weights)", 0.2, 1.0, 0.6)

    st.header("Quantum (QAOA)")
    p_layers = st.slider("QAOA layers (p)", 1, 6, 2)
    shots = st.slider("Shots", 100, 5000, 1024, step=64)
    max_iter = st.slider("Optimizer iterations", 30, 300, 80, step=10)

    run_btn = st.button("Generate & Solve", use_container_width=True)

if not run_btn:
    st.info("Configure the instance in the sidebar, then click **Generate & Solve**.")
else:
    rng = np.random.default_rng(int(seed))

    if problem == "Max-Cut":
        G = nx.generators.random_graphs.erdos_renyi_graph(n, p_edge, seed=int(seed))
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps) - 1):
                G.add_edge(list(comps[i])[0], list(comps[i + 1])[0])

        tab_solve, tab_bench = st.tabs(["Solve", "Benchmark"])

        with tab_solve:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Graph")
                st.write(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
                try:
                    st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())
                except Exception:
                    st.write("(Graphviz not found; plotting skipped.)")

            with col2:
                st.subheader("Classical Baseline")
                classical = solve_maxcut_classical(G, time_budget_s=6.0)
                st.json({
                    "best_cut_value": classical["best_value"],
                    "method": classical["method"],
                    "evaluations": classical["evaluations"],
                })
                st.caption("Partition (A side, B side):")
                st.code(str(classical["partition"]))

            with st.expander("Quantum — QAOA (simulator)", expanded=True):
                qres = qaoa_solve_maxcut(G, p=p_layers, shots=shots, max_iter=max_iter, seed=int(seed))
                st.json({
                    "qaoa_best_value": qres["best_value"],
                    "approx_ratio": qres["approx_ratio"],
                    "iterations": str(qres["iterations"]),
                })
                st.caption("Best bitstring & cut value:")
                st.code(str((qres["best_bitstring"], qres["best_value"])))

                circ = qaoa_build_param_circuit(G, p=p_layers, betas=[0.1]*p_layers, gammas=[0.1]*p_layers)
                ours_resources = estimate_resources_qiskit(circ)
                st.subheader("Resource Estimates (transpiled)")
                st.json(ours_resources)

            st.subheader("Comparison — Our QAOA vs Classiq")

            cqr_cmp = classiq_qaoa_maxcut(G, p=p_layers, shots=shots, max_iter=max_iter)
            c_res = cqr_cmp.get("resources", {}) if isinstance(cqr_cmp, dict) else {}

            rows = [
                {
                    "solver": "Ours (QAOA)",
                    "metric": "best cut",
                    "value": qres.get("best_value"),
                    "approx_ratio": qres.get("approx_ratio"),
                    "num_qubits": ours_resources.get("num_qubits"),
                    "depth": ours_resources.get("depth"),
                    "two_qubit_gates": ours_resources.get("two_qubit_gates"),
                },
                {
                    "solver": "Classiq (auto-ansatz QAOA)",
                    "metric": "energy" if cqr_cmp.get("energy") is not None else "—",
                    "value": cqr_cmp.get("energy"),
                    "approx_ratio": None,
                    "num_qubits": c_res.get("num_qubits"),
                    "depth": c_res.get("depth"),
                    "two_qubit_gates": c_res.get("two_qubit_gates"),
                },
            ]
            df_cmp = pd.DataFrame(rows)
            st.dataframe(df_cmp, use_container_width=True)

            def _fmt(x):
                return "—" if x is None else str(int(x)) if isinstance(x, (int, np.integer)) else f"{x:.3g}" if isinstance(x, (float, np.floating)) else str(x)

            def _pct_impr(lower_is_better, new, base):
                try:
                    new = float(new); base = float(base)
                    if base <= 0: return None
                    delta = (base - new) / base * 100.0 if lower_is_better else (new - base) / base * 100.0
                    return f"{delta:+.1f}%"
                except Exception:
                    return None

            depth_ours, depth_cls = ours_resources.get("depth"), c_res.get("depth")
            twoq_ours, twoq_cls = ours_resources.get("two_qubit_gates"), c_res.get("two_qubit_gates")

            depth_impr = _pct_impr(True, depth_cls, depth_ours) if (depth_ours is not None and depth_cls is not None) else None
            twoq_impr  = _pct_impr(True, twoq_cls, twoq_ours)   if (twoq_ours  is not None and twoq_cls  is not None) else None

            bullets = []
            if depth_impr:
                bullets.append(f"• **Depth** — Classiq: {_fmt(depth_cls)} vs Ours: {_fmt(depth_ours)} → {depth_impr}.")
            if twoq_impr:
                bullets.append(f"• **Two-qubit gates** — Classiq: {_fmt(twoq_cls)} vs Ours: {_fmt(twoq_ours)} → {twoq_impr}.")
            if not bullets:
                bullets.append("• Classiq synthesized and provided resource estimates. Some metrics (like op counts) may be **N/A** on this SDK build.")

            st.success("**Compare Summary**  \n" + "\n".join(bullets))

            if c_res.get("two_qubit_gates") is None:
                E = G.number_of_edges()
                est_rzz = E * p_layers
                est_cx  = 2 * E * p_layers
                st.info(
                    f"Classiq two-qubit gates not reported by SDK. "
                    f"Analytical estimate for QAOA-MaxCut (p={p_layers}): "
                    f"≈ {est_rzz} (native RZZ) or ≈ {est_cx} (via CX decomposition)."
                )

            try:
                st.markdown("**Depth comparison** (lower is better)")
                st.bar_chart(df_cmp.set_index("solver")[["depth"]])
                st.markdown("**Two-qubit gates comparison** (lower is better)")
                st.bar_chart(df_cmp.set_index("solver")[["two_qubit_gates"]])
            except Exception:
                pass

            with st.expander("Classiq — Auto-Ansatz (built-in QAOA)", expanded=False):
                st.caption("Requires: pip install classiq pyomo (no hardware).")
                try:
                    cqr = cqr_cmp  # reuse build
                    if not cqr.get("ok", False):
                        st.error(cqr.get("error", "Unknown error"))
                    else:
                        res_display = dict(cqr["resources"])
                        if res_display.get("two_qubit_gates") is None:
                            E = G.number_of_edges()
                            res_display["two_qubit_gates_est_native_rzz"] = int(E * p_layers)
                            res_display["two_qubit_gates_est_cx_equiv"] = int(2 * E * p_layers)

                        st.subheader("Synthesized circuit resources (Classiq)")
                        st.json(res_display)

                        try:
                            from qiskit.circuit import QuantumCircuit as QC
                            def _unwrap_qc(obj):
                                if isinstance(obj, (list, tuple, set, dict)):
                                    it = obj.values() if isinstance(obj, dict) else obj
                                    for item in it:
                                        hit = _unwrap_qc(item)
                                        if isinstance(hit, QC):
                                            return hit
                                    return None
                                return obj
                            qc = _unwrap_qc(cqr.get("qiskit_circuit"))
                            if isinstance(qc, QC):
                                from qiskit_aer import Aer
                                backend = Aer.get_backend("qasm_simulator")
                                result = backend.run(qc, shots=shots).result()
                                counts = result.get_counts()
                                exp_val = sample_expectation(G, counts)
                                st.write({"expected_cut_from_sampling": exp_val})
                        except Exception as e:
                            st.caption(f"Aer sampling unavailable: {e}")

                        if cqr.get("executed", False):
                            st.write({k: cqr[k] for k in ["energy", "iterations", "time_sec"] if k in cqr})
                        elif "execution_error" in cqr:
                            st.caption(f"Classiq execution skipped: {cqr['execution_error']}")
                except Exception as e:
                    import traceback
                    st.error(f"Classiq integration failed: {e}")
                    st.exception(e)
                    st.code(traceback.format_exc())

        with tab_bench:
            st.subheader("Parameter Sweep: trade-off between quality and resources")
            p_range = st.slider("p range", 1, 6, (1, 4))
            trials = st.slider("Trials per p", 1, 5, 3)
            shots_b = st.slider("Shots (benchmark)", 100, 5000, shots, step=64)
            iters_b = st.slider("Optimizer iterations (benchmark)", 20, 300, max_iter, step=10)
            run_bench = st.button("Run Benchmark", type="primary")

            if run_bench:
                p_values = list(range(p_range[0], p_range[1] + 1))
                df, classical_b = sweep_qaoa(G, p_values, shots=shots_b, max_iter=iters_b, trials=trials, seed=int(seed))
                st.caption(f"Classical baseline: {classical_b['method']} with best cut {classical_b['best_value']}")
                st.dataframe(df, use_container_width=True)
                g1 = df.groupby("p")["qaoa_best_value"].mean().reset_index()
                g2 = df.groupby("p")["two_qubit_gates"].mean().reset_index()
                g3 = df.groupby("p")["approx_ratio_vs_classical"].mean().reset_index()
                st.markdown("**Mean best cut vs p** (higher is better)")
                st.line_chart(g1, x="p", y="qaoa_best_value", height=220)
                st.markdown("**Mean two-qubit gates vs p** (lower is cheaper)")
                st.line_chart(g2, x="p", y="two_qubit_gates", height=220)
                st.markdown("**Mean approx. ratio vs classical vs p** (closer to 1 is better)")
                st.line_chart(g3, x="p", y="approx_ratio_vs_classical", height=220)

    else:
        values = rng.integers(val_min, val_max + 1, size=n).tolist()
        weights = rng.integers(wt_min, wt_max + 1, size=n).tolist()
        capacity = int(cap_ratio * sum(weights))

        st.subheader("Items")
        df_items = pd.DataFrame({"item": list(range(n)), "value": values, "weight": weights})
        st.dataframe(df_items, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Classical Baseline (DP)")
            best = solve_knapsack_dp(values, weights, capacity)
            st.json({
                "capacity": capacity,
                "best_value": best["best_value"],
                "total_weight": best["total_weight"],
                "picked_items": best["picked_items"],
                "method": "dynamic_programming",
            })

        with col2:
            with st.expander("Classiq — Auto-Ansatz (built-in QAOA)", expanded=True):
                st.caption("Uses a Pyomo knapsack model → Classiq QAOA synthesis.")
                try:
                    cqr = classiq_qaoa_knapsack(values, weights, capacity, p=p_layers, shots=shots, max_iter=max_iter)
                    if not cqr.get("ok", False):
                        st.error(cqr.get("error", "Unknown error"))
                    else:
                        st.subheader("Synthesized circuit resources (Classiq)")
                        st.json(cqr["resources"])
                        if cqr.get("executed", False):
                            st.write({k: cqr[k] for k in ["energy", "iterations", "time_sec"] if k in cqr})
                        elif "execution_error" in cqr:
                            st.caption(f"Classiq execution skipped: {cqr['execution_error']}")
                except Exception as e:
                    import traceback
                    st.error(f"Classiq integration failed: {e}")
                    st.exception(e)
                    st.code(traceback.format_exc())
