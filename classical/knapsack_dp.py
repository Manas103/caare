from typing import List, Dict

def solve_knapsack_dp(values: List[int], weights: List[int], capacity: int) -> Dict:
    """
    Knapsack via dynamic programming (O(n*capacity)).
    Returns best_value, picked_items (indices), total_weight.
    """
    n = len(values)
    W = int(capacity)
    dp = [[0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        v, wt = values[i-1], weights[i-1]
        for w in range(W+1):
            dp[i][w] = dp[i-1][w]
            if wt <= w:
                cand = dp[i-1][w-wt] + v
                if cand > dp[i][w]:
                    dp[i][w] = cand
    w = W
    picked = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            picked.append(i-1)
            w -= weights[i-1]
    picked.reverse()
    total_w = sum(weights[i] for i in picked)
    best_v = sum(values[i] for i in picked)
    return {
        "best_value": int(best_v),
        "picked_items": picked,
        "total_weight": int(total_w),
    }
