import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from qiskit import transpile
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error

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


def init_state():
    if "generated" not in st.session_state:
        st.session_state.generated = False
        st.session_state.problem = None
        st.session_state.G = None         
        st.session_state.ks_data = None 


init_state()


def generate_instance(problem, seed, **kwargs):
    """Create and store the instance in session_state. Only called when user clicks Generate."""
    rng = np.random.default_rng(int(seed))
    if problem == "Max-Cut":
        n = kwargs["n"]; p_edge = kwargs["p_edge"]
        G = nx.generators.random_graphs.erdos_renyi_graph(n, p_edge, seed=int(seed))
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps) - 1):
                G.add_edge(list(comps[i])[0], list(comps[i + 1])[0])
        st.session_state.G = G
        st.session_state.ks_data = None

    else:
        n = kwargs["n"]; val_min = kwargs["val_min"]; val_max = kwargs["val_max"]
        wt_min = kwargs["wt_min"]; wt_max = kwargs["wt_max"]; cap_ratio = kwargs["cap_ratio"]
        values = rng.integers(val_min, val_max + 1, size=n).tolist()
        weights = rng.integers(wt_min, wt_max + 1, size=n).tolist()
        capacity = int(cap_ratio * sum(weights))
        st.session_state.ks_data = (values, weights, capacity)
        st.session_state.G = None

    st.session_state.generated = True
    st.session_state.problem = problem


def require_instance(problem):
    """Guard: show a helpful message if the current sidebar selection doesn't match the stored instance."""
    if not st.session_state.generated:
        st.info("Configure the instance in the sidebar, then click **Generate & Solve**.")
        st.stop()

    if st.session_state.problem != problem:
        st.warning(
            f"You generated a **{st.session_state.problem}** instance. "
            f"Switching to **{problem}** requires pressing **Generate & Solve**."
        )
        st.stop()


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

if run_btn:
    if problem == "Max-Cut":
        generate_instance(problem, seed, n=n, p_edge=p_edge)
    else:
        generate_instance(
            problem, seed,
            n=n, val_min=val_min, val_max=val_max,
            wt_min=wt_min, wt_max=wt_max, cap_ratio=cap_ratio
        )


if problem == "Max-Cut":
    require_instance("Max-Cut")
    G = st.session_state.G

    tab_solve, tab_bench, tab_noise = st.tabs(["Solve", "Benchmark", "Noise"])

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
            bullets.append("• Classiq synthesized and provided resource estimates. Some metrics may be **N/A** on this SDK build.")
        st.success("**Compare Summary**  \n" + "\n".join(bullets))

        if c_res.get("two_qubit_gates") is None:
            E = G.number_of_edges()
            st.info(
                f"Classiq two-qubit gates not reported. QAOA-MaxCut (p={p_layers}) rough estimates: "
                f"≈ {E*p_layers} (native RZZ) or ≈ {2*E*p_layers} (via CX decomposition)."
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
                cqr = cqr_cmp
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
        run_bench = st.button("Run Benchmark", type="primary", key="run_bench")

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

    with tab_noise:
        st.subheader("Noise study (Aer depolarizing model)")

        st.markdown(
            "We first optimize angles **without noise** (fast) or optionally **re-optimize under noise** "
            "for each grid cell (slower). Then we evaluate expected cut under a depolarizing model and plot heatmaps."
        )

        # Controls
        p_values = st.multiselect("QAOA layers p", options=[1, 2, 3, 4], default=[1, 2, 3])
        shots_noise = st.slider("Shots (per point)", 256, 4096, 512, step=256)
        reopt_under_noise = st.checkbox("Re-optimize angles at each noise grid point (slower)", value=False)

        c1, c2 = st.columns(2)
        with c1:
            n_p1 = st.slider("1-qubit depolarizing grid size", 3, 9, 5)
            p1_max = st.number_input("Max 1-qubit depolarizing prob (p1)", value=0.003, min_value=0.0, step=0.001, format="%.4f")
        with c2:
            n_p2 = st.slider("2-qubit depolarizing grid size", 3, 9, 5)
            p2_max = st.number_input("Max 2-qubit depolarizing prob (p2)", value=0.02, min_value=0.0, step=0.01, format="%.3f")

        maxiter_opt = st.slider("Optimizer iters (angle fit)", 20, 200, 60, step=10)
        run_noise = st.button("Run noise sweep", type="primary", key="run_noise")

        backend = Aer.get_backend("qasm_simulator")

        def _graph_sig(G):
            edges = tuple(sorted(tuple(sorted(e)) for e in G.edges()))
            return (G.number_of_nodes(), edges)

        @st.cache_data(show_spinner=False)
        def optimize_noiseless_angles(graph_sig, p, shots, maxiter, seed):
            rng_local = np.random.default_rng(seed)
            betas0 = rng_local.uniform(0, np.pi, size=p)
            gammas0 = rng_local.uniform(0, np.pi, size=p)

            def objective(x):
                b = x[:p]; g = x[p:]
                circ = qaoa_build_param_circuit(G, p, b, g)
                tq = transpile(circ, backend)
                result = backend.run(tq, shots=shots, seed_simulator=seed).result()
                counts = result.get_counts()
                return -sample_expectation(G, counts)

            x0 = np.concatenate([betas0, gammas0])
            res = minimize(objective, x0, method="COBYLA", options=dict(maxiter=int(maxiter), rhobeg=0.4))
            return res.x[:p], res.x[p:]

        def build_noise_model(p1, p2):
            nm = NoiseModel()
            if p1 > 0:
                err1 = depolarizing_error(float(p1), 1)
                for g in ["rx", "ry", "rz", "h", "sx", "x", "id"]:
                    try: nm.add_all_qubit_quantum_error(err1, g)
                    except Exception: pass
            if p2 > 0:
                err2 = depolarizing_error(float(p2), 2)
                for g in ["cx", "cz", "rzz", "swap", "ecr"]:
                    try: nm.add_all_qubit_quantum_error(err2, g)
                    except Exception: pass
            return nm

        def eval_with_noise(p, betas, gammas, p1, p2, shots, seed):
            circ = qaoa_build_param_circuit(G, p, betas, gammas)
            nm = build_noise_model(p1, p2)
            tq = transpile(circ, backend, basis_gates=nm.basis_gates)
            result = backend.run(tq, shots=shots, noise_model=nm, seed_simulator=seed).result()
            counts = result.get_counts()
            return sample_expectation(G, counts)

        def optimize_with_noise(p, p1, p2, shots, maxiter, seed):
            rng_local = np.random.default_rng(seed + int(1e6 * (p1 + 3*p2)))
            betas0 = rng_local.uniform(0, np.pi, size=p)
            gammas0 = rng_local.uniform(0, np.pi, size=p)

            def objective(x):
                b = x[:p]; g = x[p:]
                circ = qaoa_build_param_circuit(G, p, b, g)
                nm = build_noise_model(p1, p2)
                tq = transpile(circ, backend, basis_gates=nm.basis_gates)
                result = backend.run(tq, shots=shots, noise_model=nm, seed_simulator=seed).result()
                counts = result.get_counts()
                return -sample_expectation(G, counts)

            x0 = np.concatenate([betas0, gammas0])
            res = minimize(objective, x0, method="COBYLA", options=dict(maxiter=int(maxiter), rhobeg=0.4))
            return res.x[:p], res.x[p:]

        if run_noise:
            if not p_values:
                st.warning("Pick at least one p.")
            else:
                p1_vals = np.linspace(0.0, float(p1_max), int(n_p1))
                p2_vals = np.linspace(0.0, float(p2_max), int(n_p2))

                with st.spinner("Optimizing angles and sweeping noise…"):
                    total_p = len(p_values)
                    prog_p = st.progress(0.0, text="Preparing…")

                    for idx_p, p in enumerate(p_values, start=1):
                        if reopt_under_noise:
                            st.write(f"Re-optimizing under noise for p={p} …")
                        else:
                            st.write(f"Optimizing angles (noiseless) for p={p} …")
                            b_opt, g_opt = optimize_noiseless_angles(_graph_sig(G), p, shots_noise, maxiter_opt, int(seed))

                        data = []
                        total_cells = len(p1_vals) * len(p2_vals)
                        prog_grid = st.progress(0.0, text=f"p={p}: sweeping noise grid…")
                        k = 0

                        for p2 in p2_vals:
                            for p1 in p1_vals:
                                if reopt_under_noise:
                                    b_use, g_use = optimize_with_noise(p, p1, p2, shots_noise, maxiter_opt, int(seed))
                                else:
                                    b_use, g_use = b_opt, g_opt

                                exp_cut = eval_with_noise(p, b_use, g_use, p1, p2, shots_noise, int(seed))
                                approx = exp_cut / max(1, G.number_of_edges())
                                data.append((p, float(p1), float(p2), float(exp_cut), float(approx)))

                                k += 1
                                prog_grid.progress(min(1.0, k / total_cells))

                        prog_grid.empty()

                        df_heat = pd.DataFrame(data, columns=["p", "p1_1q", "p2_2q", "expected_cut", "approx_ratio"])
                        st.write(f"Heatmap for p = {p} (expected cut). 1q prob on X, 2q prob on Y")
                        grid = df_heat.pivot(index="p2_2q", columns="p1_1q", values="expected_cut").sort_index(ascending=True)

                        fig, ax = plt.subplots(figsize=(6.0, 4.5))
                        im = ax.imshow(grid.values, aspect="auto", origin="lower")
                        ax.set_xticks(range(len(grid.columns)))
                        ax.set_xticklabels([f"{v:.3g}" for v in grid.columns], rotation=45, ha="right")
                        ax.set_yticks(range(len(grid.index)))
                        ax.set_yticklabels([f"{v:.3g}" for v in grid.index])
                        ax.set_xlabel("1-qubit depolarizing p1")
                        ax.set_ylabel("2-qubit depolarizing p2")
                        ax.set_title(f"Expected cut (p={p})")
                        fig.colorbar(im, ax=ax, shrink=0.85)
                        st.pyplot(fig)

                        prog_p.progress(min(1.0, idx_p / total_p), text=f"Completed p={p}")

                    prog_p.empty()

                st.caption(
                    "Notes: re-optimizing under noise yields more realistic performance but is slower. "
                    "This study highlights NISQ trade-offs across p and noise strengths."
                )

else:
    require_instance("Knapsack (QUBO)")
    values, weights, capacity = st.session_state.ks_data

    st.subheader("Items")
    df_items = pd.DataFrame({"item": list(range(len(values))), "value": values, "weight": weights})
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
