"""
kelly_multi.py
---------------
Replaces per-bracket independent Kelly sizing with a joint solve
across every bracket in the same event, correctly handling a MIX of
YES and NO bets sharing one bankroll constraint.

Why this needs to be more than "sum(fractions) <= 1" on a simple
partition: a YES bet on bracket i only pays off if outcome == i, but
a NO bet on bracket j pays off if outcome != j -- which overlaps with
every OTHER bracket's YES outcome. You can't treat these as a clean
mutually-exclusive set the way the single-sided (all-YES) solver from
earlier in this project could. The correct formulation evaluates
total wealth under EVERY possible outcome bin (every bracket, since
they still partition the underlying temperature distribution), and
maximizes expected log-utility summed across that whole distribution
-- which handles mixed YES/NO bets correctly because each bet's
payoff is defined per-outcome-bin, not by a single "win/lose" flag.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
from scipy.optimize import minimize


@dataclass
class Candidate:
    label: str
    side: Literal["YES", "NO"]
    model_prob: float      # P(this specific bracket is the outcome)
    cost: float             # price + fee already included (what you pay per $1 of payout)


@dataclass
class KellyResult:
    label: str
    side: str
    fraction_full_kelly: float
    fraction_after_multiplier: float
    stake_dollars: float


def _neg_expected_log_utility(fractions: np.ndarray, candidates: List[Candidate],
                                bracket_probs: np.ndarray, bracket_labels: List[str]) -> float:
    remaining_cash = 1.0 - fractions.sum()
    expected_log = 0.0

    for k, label_k in enumerate(bracket_labels):
        wealth = remaining_cash
        for f, c in zip(fractions, candidates):
            if c.side == "YES":
                # pays off only if THIS bracket is the realized outcome
                if label_k == c.label:
                    wealth += f / c.cost
            else:  # NO
                # pays off if the realized outcome is anything OTHER than this bracket
                if label_k != c.label:
                    wealth += f / c.cost
        wealth = max(wealth, 1e-9)
        expected_log += bracket_probs[k] * np.log(wealth)

    return -expected_log


def solve_joint_kelly(
    candidates: List[Candidate],
    bracket_labels: List[str],
    bracket_probs: List[float],
    kelly_multiplier: float,
    bankroll: float,
) -> List[KellyResult]:
    """
    candidates: every bracket that individually cleared MIN_EDGE_THRESHOLD
        (pre-filtered by the caller, same as before), each tagged with
        which side (YES/NO) had the edge.
    bracket_labels / bracket_probs: the FULL set of brackets in the
        event (not just the tradeable ones) with their model
        probabilities, so the objective correctly accounts for money
        lost when a non-traded bracket wins.
    """
    if not candidates:
        return []

    n = len(candidates)
    bracket_probs_arr = np.array(bracket_probs, dtype=float)
    bracket_probs_arr = bracket_probs_arr / bracket_probs_arr.sum()  # renormalize defensively

    x0 = np.full(n, 0.5 / (n + 1))
    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = [{"type": "ineq", "fun": lambda x: 1.0 - x.sum()}]  # no leverage

    result = minimize(
        _neg_expected_log_utility, x0,
        args=(candidates, bracket_probs_arr, bracket_labels),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    if not result.success:
        # Fail loudly and safely -- return zero stakes rather than
        # silently trusting an unconverged solve.
        print(f"[kelly_multi] WARNING: solver did not converge ({result.message}); "
              f"returning zero stakes for this event.")
        return [KellyResult(c.label, c.side, 0.0, 0.0, 0.0) for c in candidates]

    full_kelly = np.clip(result.x, 0.0, None)
    scaled = full_kelly * kelly_multiplier

    return [
        KellyResult(
            label=c.label, side=c.side,
            fraction_full_kelly=float(f_full), fraction_after_multiplier=float(f_scaled),
            stake_dollars=float(f_scaled * bankroll),
        )
        for c, f_full, f_scaled in zip(candidates, full_kelly, scaled)
    ]
