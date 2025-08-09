import itertools
import time
import networkx as nx

def cut_value(G: nx.Graph, bitstring):
    value = 0
    for u, v in G.edges():
        if bitstring[u] != bitstring[v]:
            value += 1
    return value

def greedy_partition(G: nx.Graph):
    n = G.number_of_nodes()
    bits = [0]*n
    improved = True
    while improved:
        improved = False
        for i in range(n):
            current = cut_value(G, bits)
            bits[i] = 1 - bits[i]
            newv = cut_value(G, bits)
            if newv <= current:
                bits[i] = 0 if bits[i] == 1 else 1
            else:
                improved = True
    return bits, cut_value(G, bits)

def brute_force(G: nx.Graph, time_budget_s=8.0):
    n = G.number_of_nodes()
    best = -1
    best_bits = None
    start = time.time()
    evals = 0
    for bits in itertools.product([0,1], repeat=n):
        evals += 1
        if time.time() - start > time_budget_s:
            break
        v = cut_value(G, bits)
        if v > best:
            best = v
            best_bits = bits
    return best_bits, best, evals

def solve_maxcut_classical(G: nx.Graph, time_budget_s=6.0):
    n = G.number_of_nodes()
    if n <= 18:
        bits, val, evals = brute_force(G, time_budget_s=time_budget_s)
        if bits is not None:
            A = {i for i, b in enumerate(bits) if b==1}
            B = set(range(n)) - A
            return dict(method="brute_force", best_value=val, partition=(A, B), evaluations=evals)
    bits, val = greedy_partition(G)
    A = {i for i, b in enumerate(bits) if b==1}
    B = set(range(n)) - A
    return dict(method="greedy", best_value=val, partition=(A, B), evaluations=None)
