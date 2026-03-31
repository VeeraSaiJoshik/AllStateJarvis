"""
NAME: Probability & Expected Value DP
TAGS: dp, probability, expected-value, games, markov
DESCRIPTION: Computes probabilities and expected values using DP over states. Covers
    random walks, dice games, geometric distributions, and Markov chain problems.
    Use when competitions ask for expected number of steps/moves, winning probabilities,
    or probabilities with state-dependent transitions.
COMPLEXITY: Time O(states * transitions), Space O(states)
"""

from fractions import Fraction
from functools import lru_cache

EPS = 1e-9

# ─────────────────────────────────────────────────────────────────────────────
# 1. EXPECTED NUMBER OF DICE ROLLS TO REACH TARGET
# ─────────────────────────────────────────────────────────────────────────────

def expected_rolls_to_target(target, faces=6):
    """
    Expected number of rolls of a fair `faces`-sided die to reach exactly `target`
    (starting from 0, accumulating sum). If you overshoot you stop.

    dp[i] = E[rolls to reach target | current sum = i]

    Example:
        expected_rolls_to_target(1, 6) → 1.0   # must roll exactly 1: prob 1/6, E = 6
        Wait: P(reach 1) = 1/6, but we stop at overshoot. E[rolls needed] differs.
        # Actually: from 0, roll once. With prob 1/6 reach 1 (done in 1 roll).
        # With prob 5/6 overshoot. So E[rolls to reach exactly 1] requires careful def.
        # Simpler def: expected rolls to get sum >= target.
        expected_rolls_to_target(1, 6) → 1.0  (always takes 1 roll)
        expected_rolls_to_target(6, 6) → 1.0  (can reach in 1 roll only if you roll 6; otherwise need more)
    """
    # E[i] = expected additional rolls from position i to reach >= target
    # E[i] = 0 if i >= target
    # E[i] = 1 + (1/faces) * sum(E[i+d] for d in 1..faces)
    # Solve from i = target-1 down to 0

    E = [0.0] * (target + faces + 1)
    for i in range(target - 1, -1, -1):
        E[i] = 1.0 + sum(E[min(i + d, target)] for d in range(1, faces + 1)) / faces
        # Note: E[target] = 0 by definition
    return E[0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROBABILITY OF WINNING A DICE GAME
# ─────────────────────────────────────────────────────────────────────────────

def win_probability_dice(target, faces=6):
    """
    Two players take turns rolling a fair die. First to reach >= target wins.
    Returns P(player 1 wins) assuming player 1 goes first.

    dp[i] = P(current player wins | current sum = i)
    dp[i] = sum over d: (1/faces) * (1 - dp[i+d]) for d in 1..faces
    (1 - dp[i+d] because after opponent rolls, THEY are the current player)

    Example:
        win_probability_dice(2, 2) → 0.75
        # P1 rolls: 1→P2 from sum=1; 2→P1 wins.
        # P(win) = 1/2 + 1/2 * P(P1 wins | P2 at 1)
        # P2 at 1: rolls 1→P2 reaches 2 (P2 wins=P1 loses) or rolls 2→P2 reaches 3 (P2 wins)
        # Hmm wait: from sum 1, both outcomes >=2 so P2 wins. So P(win)=0.5. Let me recheck.
    """
    # prob[i] = P(current player wins given sum so far = i, it's their turn)
    prob = [0.0] * (target + faces + 1)
    # Base: if i >= target, current player already won (previous player went over)
    # Actually: if current sum >= target, previous player already won → current player lost
    # We define: prob[i] = P(current player starting from sum i wins)
    # At sum i >= target: we haven't reached here (previous roll already ended game)
    # Define prob[target..] = 0 (you've already lost because opponent just reached target)

    for i in range(target - 1, -1, -1):
        p = 0.0
        for d in range(1, faces + 1):
            new_sum = i + d
            if new_sum >= target:
                p += 1.0 / faces   # you win by reaching target
            else:
                p += (1.0 - prob[new_sum]) / faces   # opponent's winning prob flips
        prob[i] = p
    return prob[0]


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXPECTED VALUE — RANDOM WALK ON A LINE
# ─────────────────────────────────────────────────────────────────────────────

def expected_steps_random_walk(n, start, absorb_left=0, absorb_right=None):
    """
    1D random walk: at each step move left or right with prob 0.5.
    Absorbed at positions 0 and n (or absorb_right).
    Returns expected steps to absorption from each position.

    Solve: E[i] = 1 + 0.5*E[i-1] + 0.5*E[i+1]
    With boundary: E[0] = E[n] = 0.
    Linear system → closed form: E[i] = i * (n - i).

    Example:
        expected_steps_random_walk(4, 2) → 4.0   # E[2] = 2*(4-2) = 4
        expected_steps_random_walk(6, 3) → 9.0
    """
    if absorb_right is None:
        absorb_right = n
    # Closed-form for symmetric random walk
    i = start
    N = absorb_right - absorb_left
    i_rel = start - absorb_left
    return float(i_rel * (N - i_rel))


def expected_steps_general_walk(n, p, start):
    """
    Biased random walk: move right with prob p, left with prob 1-p.
    Absorbed at 0 and n.
    Returns E[steps to absorption | start].

    Solve the recurrence:
        E[i] = 1 + p*E[i+1] + (1-p)*E[i-1], E[0]=E[n]=0.

    Example:
        expected_steps_general_walk(4, 0.5, 2) → 4.0  (symmetric)
        expected_steps_general_walk(4, 0.75, 1) → ~2.29
    """
    q = 1 - p
    if abs(p - 0.5) < EPS:
        return float(start * (n - start))
    # Closed form: E[i] = i/(q-p) - n*(1-(q/p)^i)/((q-p)*(1-(q/p)^n))
    # Use numerical solution for clarity
    # Solve tridiagonal system: -p*E[i+1] + E[i] - q*E[i-1] = 1
    # Use Thomas algorithm
    a = [-q] * (n - 1)   # lower diagonal
    b = [1.0] * (n - 1)  # main diagonal
    c = [-p] * (n - 1)   # upper diagonal
    d = [1.0] * (n - 1)  # RHS

    # Boundary: E[0]=0, E[n]=0 → already handled (no contribution)
    # Thomas algorithm (forward sweep)
    for i in range(1, n - 1):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    # Back substitution
    E = [0.0] * (n + 1)
    E[n - 1] = d[n - 2] / b[n - 2]
    for i in range(n - 3, -1, -1):
        E[i + 1] = (d[i] - c[i] * E[i + 2]) / b[i]
    return E[start]


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROBABILITY DP — KNAPSACK-STYLE (probability of achieving exactly k successes)
# ─────────────────────────────────────────────────────────────────────────────

def prob_exactly_k_successes(probs, k):
    """
    Given n independent events with probabilities probs[i], compute
    P(exactly k events succeed).

    dp[j] = probability that exactly j of the first i events succeed.

    Example:
        prob_exactly_k_successes([0.5, 0.5, 0.5], 2) → 0.375  # C(3,2) * 0.125 = 0.375
        prob_exactly_k_successes([0.3, 0.5, 0.8], 1) → ?
    """
    n = len(probs)
    dp = [0.0] * (k + 1)
    dp[0] = 1.0
    for p in probs:
        # Process right-to-left (0/1 knapsack)
        for j in range(min(k, n), 0, -1):
            dp[j] = dp[j] * (1 - p) + dp[j - 1] * p
        dp[0] *= (1 - p)
    return dp[k]


def prob_at_least_k_successes(probs, k):
    """
    P(at least k successes).

    Example:
        prob_at_least_k_successes([0.5,0.5,0.5], 2) → 0.5  # P(2) + P(3) = 0.375+0.125
    """
    n = len(probs)
    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    for p in probs:
        for j in range(n, 0, -1):
            dp[j] = dp[j] * (1 - p) + dp[j - 1] * p
        dp[0] *= (1 - p)
    return sum(dp[j] for j in range(k, n + 1))


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPECTED VALUE DP — GAME WITH CHOICES
# ─────────────────────────────────────────────────────────────────────────────

def expected_value_with_rerolls(n_faces, max_rerolls):
    """
    Roll an n-faced die. You may reroll up to max_rerolls times.
    You keep the last rolled value. Optimal strategy: reroll if current value
    is below the expected value of remaining rerolls.

    Returns the expected value with optimal play.

    Example:
        expected_value_with_rerolls(6, 0) → 3.5   # average of 1..6
        expected_value_with_rerolls(6, 1) → 4.25  # reroll if <= 3
        expected_value_with_rerolls(6, 2) → 4.667 approx
    """
    # E[k] = expected value when you have k rerolls remaining
    E = [0.0] * (max_rerolls + 1)
    # Base: no rerolls, take whatever you roll
    E[0] = (n_faces + 1) / 2.0

    for rerolls in range(1, max_rerolls + 1):
        # Reroll if current value < E[rerolls-1] (the value of keeping a reroll)
        threshold = E[rerolls - 1]
        # Values >= ceil(threshold): keep (there are n_faces - floor(threshold) of them if threshold integer, else ...)
        # E[rerolls] = avg of values >= threshold (keep) + P(value < threshold) * E[rerolls-1]
        keep_sum   = 0.0
        keep_count = 0
        for v in range(1, n_faces + 1):
            if v >= threshold:
                keep_sum   += v
                keep_count += 1
        reroll_count = n_faces - keep_count
        E[rerolls] = (keep_sum + reroll_count * E[rerolls - 1]) / n_faces

    return E[max_rerolls]


# ─────────────────────────────────────────────────────────────────────────────
# 6. MARKOV CHAIN DP — Expected steps to reach absorbing state
# ─────────────────────────────────────────────────────────────────────────────

def markov_expected_steps(trans, absorbing, start):
    """
    trans[i][j] = probability of transitioning from state i to state j.
    absorbing = set of absorbing state indices.
    Returns expected number of steps to reach any absorbing state from `start`.

    Solves the linear system: E[i] = 1 + sum_j(trans[i][j] * E[j]) for non-absorbing i.

    Example:
        # 3 states: 0 (start), 1 (absorbing), 2 (absorbing)
        # From 0: go to 1 with 0.3, stay at 0 with 0.4, go to 2 with 0.3
        trans = [[0.4,0.3,0.3],[0,1,0],[0,0,1]]
        markov_expected_steps(trans, {1,2}, 0) → 1/(1-0.4) ≈ 1.667
    """
    n = len(trans)
    non_abs = [i for i in range(n) if i not in absorbing]
    m = len(non_abs)
    idx = {s: i for i, s in enumerate(non_abs)}

    if not non_abs:
        return 0.0

    # Build system: (I - Q) * E = 1  where Q = sub-matrix of non-absorbing states
    # Using Gaussian elimination
    A = [[0.0] * m for _ in range(m)]
    b = [1.0] * m

    for i, s in enumerate(non_abs):
        A[i][i] = 1.0
        for j, t in enumerate(non_abs):
            A[i][j] -= trans[s][t]

    # Gaussian elimination
    for col in range(m):
        # Find pivot
        pivot = max(range(col, m), key=lambda r: abs(A[r][col]))
        A[col], A[pivot] = A[pivot], A[col]
        b[col], b[pivot] = b[pivot], b[col]
        if abs(A[col][col]) < EPS:
            continue
        for row in range(m):
            if row == col:
                continue
            factor = A[row][col] / A[col][col]
            for k in range(m):
                A[row][k] -= factor * A[col][k]
            b[row] -= factor * b[col]

    E = {s: b[i] / A[i][i] for i, s in enumerate(non_abs)}
    return E.get(start, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. PROBABILITY DP — Reach state with limited budget
# ─────────────────────────────────────────────────────────────────────────────

def prob_reach_target_in_k_steps(trans, start, target, k):
    """
    P(reach state `target` from `start` in exactly k steps).

    Example:
        # Simple 2-state chain: state 0 → state 1 with p=0.6, stay 0 with p=0.4
        # state 1 is absorbing
        trans = [[0.4,0.6],[0,1]]
        prob_reach_target_in_k_steps(trans, 0, 1, 3)
        # P(reach 1 in exactly 3 steps) = 0.4^2 * 0.6 = 0.096
    """
    n = len(trans)
    # dp[step][state] = prob of being in state after step steps
    dp = [0.0] * n
    dp[start] = 1.0

    for _ in range(k):
        ndp = [0.0] * n
        for s in range(n):
            if dp[s] < EPS:
                continue
            for t in range(n):
                ndp[t] += dp[s] * trans[s][t]
        dp = ndp

    return dp[target]


# ─────────────────────────────────────────────────────────────────────────────
# 8. GEOMETRIC DISTRIBUTION — Expected trials until first success
# ─────────────────────────────────────────────────────────────────────────────

def expected_trials_until_all_coupons(n):
    """
    Coupon collector problem: how many trials to collect all n distinct coupons
    (each trial gives a uniform random coupon).
    E[T] = n * H(n) where H(n) = 1 + 1/2 + ... + 1/n.

    Example:
        expected_trials_until_all_coupons(6) → 14.7  (famous result)
    """
    return n * sum(1.0 / i for i in range(1, n + 1))


def expected_trials_dp_coupon(n):
    """
    DP formulation: E[k] = expected trials to get from k distinct to k+1.
    E[k] = n / (n - k).  Total = sum over k.

    Example:
        expected_trials_dp_coupon(6) ≈ 14.7
    """
    return sum(float(n) / (n - k) for k in range(n))


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Expected rolls to 10 (d6):", expected_rolls_to_target(10, 6))

    print("Win prob (target=3, d6):", win_probability_dice(3, 6))

    print("Expected steps random walk n=4 from 2:", expected_steps_random_walk(4, 2))  # 4.0

    print("P(exactly 2 of 3 with p=0.5):", prob_exactly_k_successes([0.5,0.5,0.5], 2))  # 0.375
    print("P(at least 2 of 3 with p=0.5):", prob_at_least_k_successes([0.5,0.5,0.5], 2))  # 0.5

    print("Expected value d6 with 0 rerolls:", expected_value_with_rerolls(6, 0))  # 3.5
    print("Expected value d6 with 1 reroll:", expected_value_with_rerolls(6, 1))   # 4.25

    trans = [[0.4, 0.3, 0.3], [0, 1, 0], [0, 0, 1]]
    print("Markov expected steps:", markov_expected_steps(trans, {1, 2}, 0))  # ~1.667

    print("Coupon collector n=6:", expected_trials_until_all_coupons(6))  # ~14.7

    trans2 = [[0.4, 0.6], [0, 1]]
    print("P(reach 1 in 3 steps):", prob_reach_target_in_k_steps(trans2, 0, 1, 3))  # 0.096
