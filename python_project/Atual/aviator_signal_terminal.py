
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Aviator 5-state online terminal predictor
# States: -10, 0, 1, 5, 10
# The line "next~ X" means: given the history, the model's MOST PROBABLE next state is X.
# We also show a 'round' counter = number of inputs already ingested.
#
# Based on user's dataset stats and heuristics:
#   Baselines (overall):
#     p_ge5 = 0.0646  (5 or 10)
#     p_10  = 0.0184
#     p_5   = 0.0462
#     p_-10 = 0.0456
#     p_1   ≈ 0.2502
#     p_0   ≈ 0.6396
#   Regimes (two-block view detected on sample):
#     Hot:  p_ge5 ≈ 0.0851 ; slope_gap ≈ +0.00154 per step (k<=40)
#     Cold: p_ge5 ≈ 0.0579 ; slope_gap ≈ +0.00013 per step (k<=40)
#     Neutral: use overall baseline (0.0646) ; slope ≈ +0.00036 per step
#   Post -10 effect:
#     Cold regime: heavy suppression (×0.1)
#     Hot regime: moderate suppression (×0.65)
#
# USAGE:
#   $ python aviator_signal_terminal.py
#   Enter values line by line: one of {-10,0,1,5,10}
#   Commands: ':stats' (print metrics); ':reset'; ':quit'
#
# DISCLAIMER: Heuristic signal from past frequencies. No guarantees.
import sys
from collections import deque
from dataclasses import dataclass

VALID_STATES = (-10, 0, 1, 5, 10)

@dataclass
class Config:
    window: int = 400              # rolling window for regime detection
    delta_pp: float = 0.015        # +/- threshold (in absolute probability) from overall baseline to mark hot/cold
    # Overall baselines from the analyzed sample
    p_overall_ge5: float = 0.0646
    p_overall_10: float = 0.0184
    p_overall_5: float = 0.0462
    p_overall_m10: float = 0.0456
    p_overall_1: float = 0.2502
    p_overall_0: float = 0.6396
    # Regime base levels for p(>=5) and slope per gap
    p_hot_ge5: float = 0.0851
    p_cold_ge5: float = 0.0579
    slope_hot: float = 0.00154
    slope_cold: float = 0.00013
    slope_neutral: float = 0.00036
    # 10-specific
    p_overall_10_base: float = 0.0184
    slope_10: float = 0.00040
    # Post -10 modifiers for p(>=5)
    post_m10_factor_hot: float = 0.65
    post_m10_factor_cold: float = 0.10
    post_m10_factor_neutral: float = 0.35
    # Caps
    gap_cap: int = 40
    p_min: float = 1e-4
    p_max_ge5: float = 0.40

@dataclass
class State:
    hist: deque
    gap_ge5: int = -1  # steps since last >=5; -1 means "none yet"
    last: int = None

    @staticmethod
    def init(window: int):
        return State(hist=deque(maxlen=window))

def softclip(x, lo, hi):
    return max(lo, min(hi, x))

def regime_from_window(cfg: Config, hist: deque):
    # rolling p_ge5 over available history
    if not hist:
        return "neutral"
    n = len(hist)
    hits = sum(1 for v in hist if v in (5,10))
    p = hits / n
    if p >= cfg.p_overall_ge5 + cfg.delta_pp:
        return "hot"
    elif p <= cfg.p_overall_ge5 - cfg.delta_pp:
        return "cold"
    return "neutral"

def update_gap(state: State, val: int):
    if val in (5,10):
        state.gap_ge5 = 0
    else:
        state.gap_ge5 = (state.gap_ge5 + 1) if state.gap_ge5 >= 0 else -1
    state.last = val

def compute_probs(cfg: Config, state: State):
    # Determine regime
    reg = regime_from_window(cfg, state.hist)
    # Base p_ge5 and slope
    if reg == "hot":
        base = cfg.p_hot_ge5
        slope = cfg.slope_hot
        post_factor = cfg.post_m10_factor_hot
    elif reg == "cold":
        base = cfg.p_cold_ge5
        slope = cfg.slope_cold
        post_factor = cfg.post_m10_factor_cold
    else:
        base = cfg.p_overall_ge5
        slope = cfg.slope_neutral
        post_factor = cfg.post_m10_factor_neutral

    gap = state.gap_ge5 if state.gap_ge5 >= 0 else cfg.gap_cap
    p_ge5 = base + slope * min(gap, cfg.gap_cap)

    # Post -10 suppression for next-step >=5
    if state.last == -10:
        p_ge5 *= post_factor

    p_ge5 = softclip(p_ge5, cfg.p_min, cfg.p_max_ge5)

    # Model p_10 separately (rare, slight hazard slope)
    p10 = cfg.p_overall_10_base + cfg.slope_10 * min(gap, 2*cfg.gap_cap)
    p10 = softclip(p10, cfg.p_min, min(p_ge5-1e-6, 0.15))  # ensure p10 <= p_ge5 and not absurd

    p5 = max(p_ge5 - p10, cfg.p_min)

    # Allocate remaining mass to {-10, 0, 1} by baseline mix among them
    rem = 1.0 - (p10 + p5)
    rem = max(rem, 3*cfg.p_min)  # Prevent negative remainder

    # Baseline proportions among {-10,0,1}
    mass_m10 = cfg.p_overall_m10
    mass_0   = cfg.p_overall_0
    mass_1   = cfg.p_overall_1
    denom = mass_m10 + mass_0 + mass_1
    w_m10 = mass_m10/denom
    w_0   = mass_0/denom
    w_1   = mass_1/denom

    p_m10 = softclip(rem * w_m10, cfg.p_min, 0.25)
    p0    = softclip(rem * w_0, cfg.p_min, 0.95)
    p1    = max(rem - p_m10 - p0, cfg.p_min)

    # Normalize (light)
    s = p_m10 + p0 + p1 + p5 + p10
    p_m10, p0, p1, p5, p10 = [x/s for x in (p_m10, p0, p1, p5, p10)]

    probs = {-10: p_m10, 0: p0, 1: p1, 5: p5, 10: p10}
    best_state = max(probs, key=probs.get)  # "next~ X": the argmax suggestion
    return probs, best_state, reg, gap

def print_probs(probs):
    order = [-10,0,1,5,10]
    parts = []
    for k in order:
        parts.append(f"{k:>3}: {probs[k]*100:6.2f}%")
    return " | ".join(parts)

def main():
    cfg = Config()
    state = State.init(cfg.window)
    print("Aviator 5-state predictor (CLI)")
    print("Enter values {-10,0,1,5,10}; ':stats' for status, ':reset' to clear, ':quit' to exit.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            continue
        if line.startswith(":"):
            cmd = line.lower()
            if cmd in (":quit", ":q"):
                print("bye")
                break
            elif cmd == ":reset":
                state = State.init(cfg.window)
                print("state cleared.")
                continue
            elif cmd == ":stats":
                probs, best, reg, gap = compute_probs(cfg, state)
                rounds = len(state.hist)
                print(f"regime={reg:<7}  gap={gap:>3}  round={rounds:>5}  next~{best:>3}  |  " + print_probs(probs))
                continue
            else:
                print("commands: :stats  :reset  :quit")
                continue
        # parse value
        try:
            v = int(line)
        except:
            print("invalid input; enter one of: -10, 0, 1, 5, 10")
            continue
        if v not in VALID_STATES:
            print("invalid state; allowed: -10, 0, 1, 5, 10")
            continue
        # update
        state.hist.append(v)
        update_gap(state, v)
        probs, best, reg, gap = compute_probs(cfg, state)
        rounds = len(state.hist)
        print(f"regime={reg:<7}  gap={gap:>3}  round={rounds:>5}  next~{best:>3}  |  " + print_probs(probs))

if __name__ == "__main__":
    main()
