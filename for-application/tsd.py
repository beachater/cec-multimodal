from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import numpy as np


@dataclass
class Antibody:
    x: np.ndarray
    aff: float          # affinity = -f (maximize)
    I: float = 0.0      # improvement EWMA
    J: float = 0.0      # novelty score
    tag: int = 0        # short-lived tag counter
    T: int = 0          # age
    S: int = 0          # memory or selection count
    # HSAT (Hybrid Substrate-Aligned Triggering) attributes
    x_prev: np.ndarray = None   # Previous position for displacement calculation
    A_sub: float = 0.0          # Substrate alignment [-1, 1]
    A_orth: float = 0.0         # Orthogonal exploration signal [0, inf)
    C: float = 0.0              # Coherence (EWMA of alignment stability)
    displacement_norm: float = 0.0  # ||dx|| cached for efficiency


class ETFCSA_TSD:
    """
    Event-Triggered FCSA with Temporal Substrate Drift (TSD) and Rac1 forgetting.

    TSD:
      Maintain a substrate vector s in R^d.
      On each accepted improvement with delta dx = x_new - x_old, update s += eta * dx.
      All variation is applied in drifted coordinates: x_adj = x - lambda_s * s, mutate x_adj,
      then map back: x_new = clip(x_adj + lambda_s * s).
      Periodic decay: s *= rho every drift_interval evaluations.

    Rac1:
      Activity A = T / (S + eps) is used to
        1) penalize event score via exp(-gamma_rac1 * max(0, A - c_threshold))
        2) raise a mutation floor when A exceeds c_threshold
        3) reseed stale high-activity untagged individuals each tick

    Extras kept minimal:
      tiny IICO-like spark
      opposition-biased reseed with elementwise min/max bounds
      short coordinate polish at end
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        N: int = 60,
        n_select: int = 12,
        n_clones: int = 4,
        r: float = 2.0,
        a_frac: float = 0.12,
        seed: Optional[int] = 42,
        max_evals: int = 350_000,
        fire_target: float = 0.2,
        alpha_I: float = 1.0,
        beta_J: float = 0.5,
        threshold_eta: float = 0.05,
        spark_prob: float = 0.04,
        tag_half_life: int = 250,
        clearance_period: float = 0.06,
        budget_per_tick: int = 200,
        c_threshold: float = 4.0,   # More lenient before RAC1 penalty (was 3.0)
        gamma_rac1: float = 0.5,    # Softer penalty (was 0.75)
        # TSD params (adjusted for HSAT)
        eta: float = 0.15,           # substrate learning rate (reduced for stability)
        lambda_s: float = 0.5,       # strength of coordinate shift
        rho: float = 0.95,           # decay factor for substrate (faster adaptation)
        drift_interval: int = 2000,  # evals between substrate decay
        # HSAT (Hybrid Substrate-Aligned Triggering) params
        alpha_min: float = 0.3,      # Early: 30% exploit, 70% explore
        alpha_max: float = 0.8,      # Late: 80% exploit, 20% explore
        tau_alpha: float = 0.4,      # Transition midpoint
        beta_coherence: float = 0.9, # Coherence smoothing factor
        gamma_base: float = 0.4,     # Weight for existing I/J signals (vs HSAT)
        exploit_weight: float = 1.5, # Substrate-aligned amplification
        explore_weight: float = 2.0, # Orthogonal exploration bonus
        explore_quota: float = 0.15, # 15% guaranteed explorers
        progress: Optional[Callable[[int], None]] = None,
    ):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.N = int(N)
        self.n_select = int(n_select)
        self.n_clones = int(n_clones)
        self.r = float(r)
        self.a_frac = float(a_frac)
        self.max_evals = int(max_evals)

        self.fire_target = float(fire_target)
        self.alpha_I = float(alpha_I)
        self.beta_J = float(beta_J)
        self.threshold_eta = float(threshold_eta)

        self.spark_prob = float(spark_prob)
        self.tag_half_life = int(tag_half_life)
        self.clear_every = max(1, int(clearance_period * self.max_evals))
        self.budget_per_tick = int(budget_per_tick)

        self.c_threshold = float(c_threshold)
        self.gamma_rac1 = float(gamma_rac1)

        self.eta = float(eta)
        self.lambda_s = float(lambda_s)
        self.rho = float(rho)
        self.drift_interval = int(drift_interval)

        # HSAT parameters
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.tau_alpha = float(tau_alpha)
        self.beta_coherence = float(beta_coherence)
        self.gamma_base = float(gamma_base)
        self.exploit_weight = float(exploit_weight)
        self.explore_weight = float(explore_weight)
        self.explore_quota = float(explore_quota)

        # generation cap (number of outer loop iterations)
        # Set high enough to never be the limiting factor - eval budget takes precedence
        # Each generation uses at minimum a few evals, so max_evals is a safe upper bound
        # This ensures all 60 environments in DMMOP are processed (1.5M evals for D=5)
        self.max_gens = self.max_evals  # Will stop on eval budget, not gen count

        self.rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.widths = self.bounds[:, 1] - self.bounds[:, 0]
        self.a_vec = self.a_frac * self.widths

        self._progress = progress
        self.evals = 0

        self.pop: List[Antibody] = []
        self.best_x: Optional[np.ndarray] = None
        self.best_f: float = float("inf")
        self.history: List[float] = []
        self.theta = 0.0
        self._grid_idx: List[int] = []

        # Temporal Substrate Drift state
        self.s = np.zeros(self.dim, dtype=float)
        self._last_decay_eval = 0

    # evaluation wrapper
    def _objective(self, x: np.ndarray) -> float:
        self.evals += 1
        # do not call per-evaluation progress here; report per-generation instead
        # substrate decay on schedule
        if self.evals - self._last_decay_eval >= self.drift_interval:
            self.s *= self.rho
            self._last_decay_eval = self.evals
        return float(self.func(x))

    # init
    def _init(self):
        self.pop.clear()
        for _ in range(self.N):
            x = self.bounds[:, 0] + self.rng.random(self.dim) * self.widths
            f = self._objective(x)
            ab = Antibody(x=x, aff=-f)
            # Initialize HSAT attributes
            ab.x_prev = x.copy()
            ab.A_sub = 0.0
            ab.A_orth = 1.0  # High exploration score initially
            ab.C = 0.0
            ab.displacement_norm = 0.0
            self.pop.append(ab)
            if f < self.best_f:
                self.best_f, self.best_x = f, x.copy()
        self._grid_idx = list(range(min(self.N, max(8, self.dim))))
        self.s[:] = 0.0
        self._last_decay_eval = self.evals

    # signals
    def _update_signals(self, idx: int, old_aff: float, x_old: np.ndarray = None):
        ab = self.pop[idx]
        gain = max(0.0, ab.aff - old_aff)
        ab.I = 0.85 * ab.I + 0.15 * gain
        if self._grid_idx:
            xs = np.stack([self.pop[j].x for j in self._grid_idx], axis=0)
            d = np.linalg.norm(xs - ab.x[None, :], axis=1)
            if d.size:
                ab.J = float(np.min(d) / (np.mean(d) + 1e-12))
        if ab.tag > 0:
            ab.tag -= 1

        # HSAT: Compute substrate alignment signals
        if x_old is not None:
            dx = ab.x - x_old
            dx_norm = np.linalg.norm(dx)
            ab.displacement_norm = dx_norm
            s_norm = np.linalg.norm(self.s)

            if dx_norm > 1e-12 and s_norm > 1e-12:
                s_hat = self.s / s_norm
                ab.A_sub = np.dot(dx, s_hat) / dx_norm
                ab.A_orth = np.sqrt(max(0, 1 - ab.A_sub**2)) * dx_norm
            else:
                ab.A_sub = 0.0
                ab.A_orth = 0.0

            # Update coherence (EWMA of alignment stability)
            ab.C = self.beta_coherence * ab.C + (1 - self.beta_coherence) * abs(ab.A_sub)

        # Store current position as previous for next iteration
        ab.x_prev = ab.x.copy()

    # TSD coordinate helpers
    def _to_drifted(self, x: np.ndarray) -> np.ndarray:
        return x - self.lambda_s * self.s

    def _from_drifted(self, x_adj: np.ndarray) -> np.ndarray:
        y = x_adj + self.lambda_s * self.s
        np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
        return y

    # HSAT: Dynamic alpha for exploration/exploitation balance
    def _compute_dynamic_alpha(self) -> float:
        """
        Compute exploration/exploitation balance based on optimization progress.

        Returns alpha in [alpha_min, alpha_max]:
        - Early optimization (progress~0): alpha ≈ alpha_min (more exploration)
        - Late optimization (progress~1): alpha ≈ alpha_max (more exploitation)

        The transition follows an exponential curve controlled by tau_alpha.
        """
        progress = self.evals / self.max_evals
        return self.alpha_min + (self.alpha_max - self.alpha_min) * (1 - math.exp(-progress / self.tau_alpha))

    # FCSA mutate with Rac1 floor, performed in drifted coordinates
    def _mutate_fcsa(self, x: np.ndarray, a_norm: float, A: float) -> np.ndarray:
        # Clamp exponent to prevent overflow (exp(x) overflows around x > 709)
        exponent = min(700.0, -self.r * a_norm)
        p_base = math.exp(exponent)
        over = max(0.0, A - self.c_threshold)
        p_floor = min(0.9, 0.2 + 0.15 * over)
        p = max(p_base, p_floor)

        x_adj = self._to_drifted(x)
        mask = self.rng.random(self.dim) < p
        if np.any(mask):
            step = self.rng.uniform(-self.a_vec, self.a_vec)
            x_adj2 = x_adj + mask.astype(float) * step
        else:
            x_adj2 = x_adj
        return self._from_drifted(x_adj2)

    # spark in drifted coordinates
    def _spark(self, x: np.ndarray) -> np.ndarray:
        x_adj = self._to_drifted(x)
        z = self.rng.random(self.dim)
        z = 3.99 * z * (1 - z)
        kick = (z - 0.5) * 0.5 * self.widths
        y_adj = x_adj + kick
        np.clip(y_adj, self.bounds[:, 0] - self.lambda_s * self.s,
                self.bounds[:, 1] - self.lambda_s * self.s, out=y_adj)
        y = self._from_drifted(y_adj)
        opp = self.bounds[:, 0] + self.bounds[:, 1] - y
        mid = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
        lo = np.minimum(mid, opp)
        hi = np.maximum(mid, opp)
        y = self.rng.uniform(lo, hi)
        np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
        return y

    # improvement acceptance with TSD update
    def _accept_if_better(self, ab: Antibody, x_old: np.ndarray, f_new: float, x_new: np.ndarray):
        improved = f_new < (-ab.aff)
        if improved:
            dx = x_new - x_old
            ab.x_prev = x_old  # Track for HSAT alignment computation
            ab.x = x_new
            ab.aff = -f_new
            ab.S = max(1, ab.S + 1)
            ab.T = 0
            ab.tag = max(ab.tag, self.tag_half_life)
            # Temporal Substrate Drift update
            self.s += self.eta * dx
            if f_new < self.best_f:
                self.best_f, self.best_x = f_new, x_new.copy()
        return improved

    # one individual fire
    def _fire_one(self, i: int) -> int:
        ab = self.pop[i]
        old_aff = ab.aff
        x_old = ab.x.copy()  # Store for HSAT alignment tracking
        A = ab.T / (ab.S + 1e-12)

        if self.rng.random() < self.spark_prob:
            cand = self._spark(ab.x)
        else:
            a_norm = 0.5
            cand = self._mutate_fcsa(ab.x, a_norm, A)

        f = self._objective(cand)
        self._accept_if_better(ab, x_old, f, cand)
        self._update_signals(i, old_aff, x_old)  # Pass x_old for alignment
        return 1

    # micro cloning on hottest few, with TSD in acceptance
    def _micro_clone(self, hot_indices: List[int], budget: int) -> int:
        if budget <= 0 or not hot_indices:
            return 0
        used = 0
        hot = sorted(hot_indices, key=lambda j: self.pop[j].I, reverse=True)[:min(len(hot_indices), self.n_select)]
        affs = np.array([self.pop[j].aff for j in hot])
        a_min, a_max = float(affs.min()), float(affs.max())
        denom = max(a_max - a_min, 1e-12)

        for j in hot:
            if used >= budget:
                break
            ab = self.pop[j]
            A = ab.T / (ab.S + 1e-12)
            a_norm = (ab.aff - a_min) / denom if denom > 0 else 0.5
            k = max(1, int(round(1 + a_norm * (self.n_clones - 1))))
            for _ in range(k):
                if used >= budget: break
                y = self._mutate_fcsa(ab.x, a_norm, A)
                f = self._objective(y); used += 1
                self._accept_if_better(ab, ab.x.copy(), f, y)
        return used

    # Rac1 reseed of zombies
    def _rac1_reseed(self):
        new_pop: List[Antibody] = []
        reseed_count = 0
        for ab in self.pop:
            improving = ab.I > 1e-12
            A = ab.T / (ab.S + 1e-12)
            if (A > self.c_threshold) and (not improving) and (ab.tag == 0):
                reseed_count += 1
                # MULTIMODAL: 80% random, 20% opposition-based for diversity
                if self.best_x is None or self.rng.random() < 0.8:
                    # Pure random - explore new regions
                    y = self.bounds[:, 0] + self.rng.random(self.dim) * self.widths
                else:
                    # opposition near best in drifted sense
                    best_drifted = self._to_drifted(self.best_x)
                    opp = self.bounds[:, 0] + self.bounds[:, 1] - (best_drifted + self.lambda_s * self.s)
                    mid = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
                    lo = np.minimum(mid, opp)
                    hi = np.maximum(mid, opp)
                    y = self.rng.uniform(lo, hi)
                np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
                f = self._objective(y)
                ab = Antibody(x=y, aff=-f, tag=self.tag_half_life, T=0, S=0)
                # Initialize HSAT attributes for new antibody
                ab.x_prev = y.copy()
                ab.A_sub = 0.0
                ab.A_orth = 1.0  # High exploration initially
                ab.C = 0.0
                ab.displacement_norm = 0.0
                if f < self.best_f:
                    self.best_f, self.best_x = f, y.copy()
            new_pop.append(ab)
        self.pop = new_pop

    # clearance
    def _clearance(self):
        survivors: List[Antibody] = []
        for ab in self.pop:
            if ab.tag > 0 or ab.I > 1e-12:
                survivors.append(ab)
        need = self.N - len(survivors)
        for _ in range(need):
            # MULTIMODAL: 80% random, 20% opposition-based for diversity
            if self.best_x is None or self.rng.random() < 0.8:
                # Pure random - explore new regions
                y = self.bounds[:, 0] + self.rng.random(self.dim) * self.widths
            else:
                opp = self.bounds[:, 0] + self.bounds[:, 1] - self.best_x
                mid = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
                lo = np.minimum(mid, opp)
                hi = np.maximum(mid, opp)
                y = self.rng.uniform(lo, hi)
            np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
            f = self._objective(y)
            ab = Antibody(x=y, aff=-f)
            # Initialize HSAT attributes for new antibody
            ab.x_prev = y.copy()
            ab.A_sub = 0.0
            ab.A_orth = 1.0  # High exploration initially
            ab.C = 0.0
            ab.displacement_norm = 0.0
            survivors.append(ab)
            if f < self.best_f:
                self.best_f, self.best_x = f, y.copy()
        self.pop = survivors

    # HSAT: Hybrid Substrate-Aligned Event Triggering
    def _pick_indices(self) -> List[int]:
        """
        Hybrid Substrate-Aligned Event Triggering (HSAT).

        Balances:
        1. Substrate-aligned exploitation (fire individuals moving with learned direction)
        2. Orthogonal exploration (fire individuals exploring new directions)
        3. Existing I/J signals for backward compatibility
        4. RAC1 forgetting penalty
        """
        # Dynamic alpha based on progress
        alpha_t = self._compute_dynamic_alpha()

        # Substrate statistics
        s_norm = np.linalg.norm(self.s)
        s_hat = self.s / (s_norm + 1e-12) if s_norm > 1e-12 else np.zeros(self.dim)

        # Mean displacement for normalization
        disps = [ab.displacement_norm for ab in self.pop if ab.displacement_norm > 0]
        mean_disp = np.mean(disps) if disps else 1.0

        scores = []
        explore_scores = []

        for i, ab in enumerate(self.pop):
            ab.T += 1  # Age increment

            # Compute alignment if we have previous position
            if ab.x_prev is not None:
                dx = ab.x - ab.x_prev
                dx_norm = np.linalg.norm(dx)

                if dx_norm > 1e-12 and s_norm > 1e-12:
                    ab.A_sub = np.dot(dx, s_hat) / dx_norm
                    ab.A_orth = np.sqrt(max(0, 1 - ab.A_sub**2)) * dx_norm / (mean_disp + 1e-12)
                else:
                    ab.A_sub = 0.0
                    ab.A_orth = 0.0
            else:
                ab.A_sub = 0.0
                ab.A_orth = 1.0  # New individuals get exploration credit

            # Update coherence
            ab.C = self.beta_coherence * ab.C + (1 - self.beta_coherence) * abs(ab.A_sub)

            # Component scores
            S_exploit = ab.A_sub * ab.C * self.exploit_weight
            S_explore = ab.A_orth * self.explore_weight
            explore_scores.append(S_explore)

            # Hybrid score
            S_hybrid = alpha_t * S_exploit + (1 - alpha_t) * S_explore

            # Base score (existing signals)
            S_base = self.alpha_I * ab.I + self.beta_J * ab.J

            # Combined score
            S_combined = self.gamma_base * S_base + (1 - self.gamma_base) * S_hybrid

            # RAC1 penalty (existing)
            A = ab.T / (ab.S + 1e-12)
            over = max(0.0, A - self.c_threshold)
            penalty = math.exp(-self.gamma_rac1 * over)

            scores.append(S_combined * penalty)

        # Select by score
        order = np.argsort(scores)[::-1]
        hot = [int(i) for i in order if scores[i] > self.theta]

        # CRITICAL: Enforce exploration quota to prevent echo chamber
        n_explore_min = max(2, int(self.N * self.explore_quota))
        explore_order = np.argsort(explore_scores)[::-1]

        explorers_added = 0
        for i in explore_order:
            if int(i) not in hot and explorers_added < n_explore_min:
                if self.pop[i].T < 500:  # Not completely stale
                    hot.append(int(i))
                    explorers_added += 1

        # Minimum population coverage
        if len(hot) < max(2, self.N // 10):
            extra = self.rng.choice(self.N, size=min(self.N // 10, self.N), replace=False)
            hot = list(dict.fromkeys(list(hot) + list(map(int, extra))))

        # Update x_prev for fired individuals
        for i in hot:
            self.pop[i].x_prev = self.pop[i].x.copy()

        return hot

    # final polish
    def _polish(self, steps: int = 120):
        if self.best_x is None:
            return
        x = self.best_x.copy()
        f = self.best_f
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        step = 0.04 * self.widths
        for t in range(steps):
            d = t % self.dim
            for sgn in (+1.0, -1.0):
                cand = x.copy()
                cand[d] = np.clip(cand[d] + sgn * step[d], lo[d], hi[d])
                fc = self._objective(cand)
                if fc < f:
                    x, f = cand, fc
            step *= 0.985
            if np.max(step) < 1e-12:
                break
        if f < self.best_f:
            self.best_f, self.best_x = f, x.copy()

    # main
    def optimize(self, progress: Callable[..., None] | None = None):
    # allow external progress callback to override constructor arg
        if progress is not None:
            self._progress = progress

        self._init()

        # ---- emit gen=0 snapshot (initial population) ----
        if self._progress is not None:
            try:
                pop_arr = np.stack([ab.x for ab in self.pop]).astype(float) if self.pop else np.empty((0, self.dim))
                fit_arr = np.array([-ab.aff for ab in self.pop], dtype=float)  # objective values
                self._progress(
                    gen=0,
                    pop=pop_arr,
                    fitness=fit_arr,
                    best_fitness=self.best_f,
                    gbest=self.best_x.copy() if self.best_x is not None else None,
                    evals=self.evals,
                )
            except Exception:
                pass

        last_clear = 0
        gen = 0

        # iterate in ticks (generations); stop when either eval budget exhausted or gen reaches max_gens
        while self.evals < self.max_evals and gen < self.max_gens:
            budget = min(self.budget_per_tick, self.max_evals - self.evals)

            hot = self._pick_indices()
            if not hot:
                self.theta *= 0.9
                # still advance generation counter to keep curves moving
                gen += 1
                continue

            b1 = max(1, budget // 2)
            b2 = budget - b1

            used = 0
            for i in hot:
                if used >= b1:
                    break
                used += self._fire_one(i)

            k = max(1, len(hot) // 3)
            used += self._micro_clone(hot[:k], b2)

            fired_fraction = len(hot) / max(1, self.N)
            self.theta += self.threshold_eta * (fired_fraction - self.fire_target)

            self._rac1_reseed()

            if self.evals - last_clear >= self.clear_every:
                self._clearance()
                last_clear = self.evals

            # track best fitness per generation
            self.history.append(self.best_f)

            # per-generation progress (use gen+1 so it follows the gen=0 snapshot)
            if self._progress is not None:
                try:
                    pop_arr = np.stack([ab.x for ab in self.pop]) if len(self.pop) > 0 else np.empty((0, self.dim))
                    fit_arr = np.array([-ab.aff for ab in self.pop], dtype=float)
                    self._progress(
                        gen=gen + 1,
                        pop=pop_arr,
                        fitness=fit_arr,
                        best_fitness=self.best_f,
                        gbest=self.best_x.copy() if self.best_x is not None else None,
                        evals=self.evals,
                    )
                except Exception:
                    pass

            gen += 1
            if self.evals >= self.max_evals:
                break
            if gen >= self.max_gens:
                break

        self._polish()

        # enforce history length cap
        if len(self.history) > self.max_gens:
            self.history = self.history[: self.max_gens]

        return self.best_x.copy(), self.best_f, {
            "evals_used": self.evals,
            "generations_run": len(self.history),
            "history": self.history,
            "substrate_norm": float(np.linalg.norm(self.s)),
        }



# demo
if __name__ == "__main__":
    def ackley(x: np.ndarray) -> float:
        a, b, c = 20.0, 0.2, 2 * np.pi
        d = x.size
        sum_sq = np.sum(x * x)
        sum_cos = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.e

    dim = 2
    bounds = [(-5.0, 5.0)] * dim

    opt = ETFCSA_TSD(
        func=ackley,
        bounds=bounds,
        N=60,
        max_evals=50_000,
        seed=123,
        budget_per_tick=150,
        eta=0.25,
        lambda_s=0.5,
        rho=0.985,
        drift_interval=1500
    )
    xbest, fbest, info = opt.optimize()
    print("best f", fbest, "evals", info["evals_used"], "||s||", info["substrate_norm"])
