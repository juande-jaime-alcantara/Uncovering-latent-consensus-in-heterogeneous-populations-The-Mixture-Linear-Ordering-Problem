#!/usr/bin/env python3
import re
import os
import time
import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import gurobipy as gp
from gurobipy import GRB

# ============================================================
# Global Configuration & Default Parameters
# ============================================================

# Base directory of the script. Used as an anchor to safely resolve 
# relative paths regardless of where the script is executed from.
BASE_DIR = Path(__file__).resolve().parent

# --- File Paths ---
# Default instance to solve if no datafile is provided (e.g., in PyCharm).
DEFAULT_DATAFILE = BASE_DIR / "../inputs/R11.dat"

# Path to store the detailed, human-readable solution report.
DEFAULT_OUTFILE = BASE_DIR / "../outputs/solutionsMLOP_heu.txt"

# Path to append a one-line performance summary of the execution.
DEFAULT_SUMMARYFILE = BASE_DIR / "../outputs/summaryMLOP_heu.txt"

# --- Heuristic & Optimization Parameters ---
# Default number of heterogeneous groups (g) to partition the population into.
DEFAULT_NUM_GRUPOS = 2

# Maximum global runtime in seconds for the entire multi-start heuristic.
DEFAULT_TIME_LIMIT = 20000.00

# Maximum number of Alternating Minimization iterations per run.
DEFAULT_IT_TOT = 12

# Number of multi-start runs (different initial random weights/permutations).
DEFAULT_RESTARTS = 20

# Random seed for reproducibility of initial starting points.
DEFAULT_SEED = 123

# --- Output Formatting ---
# Controls how the reconstructed Y variables are printed in the output file.
# To see the strictly normalized continuous variables [0, 1] as defined in 
# the mathematical model (Lemma 1), use PRINT_SCALE = 1 and PRINT_INT = False.
PRINT_SCALE = 1
PRINT_INT = False


# ============================================================
# .dat PARSER (n, nswaps, a). IGNORES g from the .dat.
# ============================================================
def _parse_scalar_int(text: str, key: str) -> Optional[int]:
    """Extracts a scalar integer value (like 'n: 10') from the raw text."""
    m = re.search(rf"\b{re.escape(key)}\s*:\s*(-?\d+)\b", text)
    return int(m.group(1)) if m else None

def _parse_matrix_a(text: str, n: int) -> List[List[int]]:
    """
    Extracts the matrix 'a' from the text file. 
    Assumes a 1-based indexing structure of size (n+1)x(n+1) for convenience,
    meaning row 0 and column 0 are ignored.
    """
    m = re.search(r"\ba\s*:\s*\[\s*(.*?)\s*\]", text, flags=re.DOTALL | re.MULTILINE)
    if not m:
        raise ValueError("Cannot find block 'a: [ ... ]' in the .dat")

    nums = re.findall(r"-?\d+", m.group(1))
    vals = [int(x) for x in nums]
    if len(vals) != n * n:
        raise ValueError(f"In 'a' expected {n*n} integers but found {len(vals)}.")

    a = [[0] * (n + 1) for _ in range(n + 1)]
    k = 0
    for r in range(1, n + 1):
        for s in range(1, n + 1):
            a[r][s] = vals[k]
            k += 1
    return a

def load_instance(path: Path) -> Tuple[int, Optional[int], List[List[int]]]:
    """Loads and parses the .dat file returning the number of items and the preference matrix."""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    n = _parse_scalar_int(txt, "n")
    if n is None:
        raise ValueError("Cannot find 'n' in the .dat")

    nswaps = _parse_scalar_int(txt, "nswaps")
    a = _parse_matrix_a(txt, n)
    return n, nswaps, a

def build_costs(a: List[List[int]], n: int) -> List[List[float]]:
    """
    Builds the normalized cost/preference matrix 'c' from 'a'.
    c[r][s] = a[r][s] / (a[r][s] + a[s][r]). If the sum is 0, defaults to 0.5.
    Uses 1-based indexing.
    """
    c = [[0.0] * (n + 1) for _ in range(n + 1)]
    for r in range(1, n + 1):
        for s in range(1, n + 1):
            if r == s:
                continue
            denom = a[r][s] + a[s][r]
            c[r][s] = 0.5 if denom == 0 else a[r][s] / denom
    return c


# ============================================================
# SLURM Threads helper
# ============================================================
def _set_threads_from_slurm(model: gp.Model):
    """Dynamically sets Gurobi threads if executed within a SLURM cluster job."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        model.setParam("Threads", int(slurm_cpus))


# ============================================================
# Helper: query xi for any (r,s) using only r<s keys
# ============================================================
def pref_xi(xi: Dict[Tuple[int, int, int], int], r: int, s: int, i: int) -> int:
    """
    Returns 1 if item r is preferred over s in group i (r > s), 0 otherwise.
    To halve memory/variables, xi ONLY stores keys where min < max.
    This function deduces the inverse preference when r > s.
    """
    if r == s:
        return 0
    if r < s:
        return int(xi.get((r, s, i), 0))
    else:
        return 1 - int(xi.get((s, r, i), 0))


# ============================================================
# Ranking from xi
# ============================================================
def extract_ranking_from_xi(n: int, g: int, xi: Dict[Tuple[int, int, int], int], i: int) -> List[int]:
    """
    Reconstructs the 1D ranking permutation from the pairwise binary variables xi
    by counting the total number of 'wins' (Borda score) for each item.
    """
    items = range(1, n + 1)
    score = {r: sum(pref_xi(xi, r, s, i) for s in items if s != r) for r in items}
    # Sort primarily by score (descending). Tie-breaker by item index.
    return sorted(items, key=lambda r: (-score[r], r))


# ============================================================
# Objective Evaluation
# ============================================================
def evaluate_solution(n: int, g: int, c: List[List[float]], xi: dict, wi: List[float]) -> float:
    """
    Evaluates the objective function value for a fixed set of weights (wi) 
    and pairwise preferences (xi). Measures the absolute deviation from the consensus.
    """
    obj = 0.0
    items = range(1, n + 1)
    groups = range(1, g + 1)
    
    for r in items:
        for s in items:
            if r < s:
                cap = sum(wi[i] * xi.get((r, s, i), 0) for i in groups)
                obj += abs(c[r][s] - cap)
    return obj


# ============================================================
# Construction of y according to Lemma 1 (min-structure)
# ============================================================
def build_y_min_structure(n: int, g: int, c: List[List[float]],
                          xi: Dict[Tuple[int, int, int], int],
                          wi: List[float],
                          eps: float = 1e-12) -> Dict[Tuple[int, int, int], float]:
    """
    Reconstructs the optimal continuous 'y' matrix variables directly from 
    'xi' and 'wi' using the analytical min-structure proven in Lemma 1.
    This guarantees perfectly normalized fractions that satisfy constraints 
    without needing to feed them back to the MIP solver.
    """
    items = range(1, n + 1)
    groups = range(1, g + 1)

    y: Dict[Tuple[int, int, int], float] = {(r, s, i): 0.0 for r in items for s in items for i in groups if r != s}

    for r in items:
        for s in items:
            if r >= s:
                continue

            crs = c[r][s]
            csr = c[s][r]

            A = [i for i in groups if pref_xi(xi, r, s, i) == 1]
            B = [i for i in groups if i not in A]

            WA = sum(wi[i] for i in A)
            WB = sum(wi[i] for i in B)

            # Case 1 from Lemma 1
            if crs <= WA + eps:
                if WA <= eps:
                    for i in groups:
                        y[(r, s, i)] = 0.0
                        y[(s, r, i)] = wi[i]
                else:
                    alpha = crs / WA
                    for i in groups:
                        y_rs = alpha * wi[i] if i in A else 0.0
                        y[(r, s, i)] = y_rs
                        y[(s, r, i)] = wi[i] - y_rs
            
            # Case 2 from Lemma 1
            else:
                if WB <= eps:
                    for i in groups:
                        y_rs = wi[i] if i in A else 0.0
                        y[(r, s, i)] = y_rs
                        y[(s, r, i)] = wi[i] - y_rs
                else:
                    alpha = csr / WB
                    for i in groups:
                        y_sr = alpha * wi[i] if i in B else 0.0
                        y_rs = wi[i] - y_sr
                        y[(r, s, i)] = y_rs
                        y[(s, r, i)] = y_sr

    return y


# ============================================================
# PHASE 1: Fixed wi -> Optimize xi (Alternating Minimization)
# ============================================================
def phase_1(n: int, g: int, c: List[List[float]], wi: List[float], xi_start=None,
            verbose: int = 0, timelimit: Optional[float] = None,
            env: Optional[gp.Env] = None):
    """
    First half of the Alternating Minimization heuristic.
    Keeps group weights (wi) fixed as parameters and optimizes the pairwise
    rankings (xi) to minimize the deviation objective.
    Uses Gurobi to solve the resulting simplified Integer Program.
    """
    items = range(1, n + 1)
    groups = range(1, g + 1)

    m = gp.Model("PHASE_1", env=env) if env else gp.Model("PHASE_1")

    try:
        m.setParam("MIPGap", 1e-5)
        m.setParam("OutputFlag", 1 if verbose else 0)
        _set_threads_from_slurm(m)
        if timelimit is not None:
            m.setParam("TimeLimit", float(timelimit))

        # Binary ranking variables
        x = {(r, s, i): m.addVar(vtype=GRB.BINARY)
             for r in items for s in items for i in groups if r < s}

        # Absolute deviation variables
        v = {(r, s): m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
             for r in items for s in items if r < s}

        m.setObjective(gp.quicksum(v[(r, s)] for r in items for s in items if r < s), GRB.MINIMIZE)

        # Transitivity constraints to ensure xi forms a valid permutation
        for i in groups:
            for r in items:
                for s in items:
                    if s <= r: continue
                    for t in items:
                        if t <= s: continue
                        m.addConstr(x[(r, s, i)] + x[(s, t, i)] - x[(r, t, i)] >= 0)
                        m.addConstr(x[(r, s, i)] + x[(s, t, i)] - x[(r, t, i)] <= 1)

        # Linearized absolute value constraints
        for r in items:
            for s in items:
                if r < s:
                    cap = gp.quicksum(wi[i] * x[(r, s, i)] for i in groups)
                    m.addConstr(v[(r, s)] >= c[r][s] - cap)
                    m.addConstr(v[(r, s)] >= cap - c[r][s])

        # Warm start using the previous iteration's best solution
        if xi_start is not None:
            for (r, s, i), val in xi_start.items():
                if (r, s, i) in x:
                    x[(r, s, i)].Start = float(round(val))

            for r in items:
                for s in items:
                    if r < s:
                        cap0 = sum(wi[i] * (1 if xi_start.get((r, s, i), 0) else 0) for i in groups)
                        v[(r, s)].Start = abs(c[r][s] - cap0)

        m.optimize()

        if m.SolCount > 0:
            xi = {(r, s, i): int(round(x[(r, s, i)].X)) for (r, s, i) in x.keys()}
            return xi, m.ObjVal
        else:
            return None, None

    finally:
        m.dispose()


# ============================================================
# PHASE 2: Fixed xi -> Optimize wi (Alternating Minimization)
# ============================================================
def phase_2(n: int, g: int, c: List[List[float]], xi: dict, wi_start=None,
            verbose: int = 0, timelimit: Optional[float] = None,
            env: Optional[gp.Env] = None):
    """
    Second half of the Alternating Minimization heuristic.
    Keeps the rankings (xi) fixed as parameters and optimizes the group
    weights (wi). Because xi are fixed, this becomes a pure continuous 
    Linear Program (LP) that solves extremely fast.
    """
    items = range(1, n + 1)
    groups = range(1, g + 1)

    m = gp.Model("PHASE_2", env=env) if env else gp.Model("PHASE_2")

    try:
        m.setParam("MIPGap", 1e-5)
        m.setParam("OutputFlag", 1 if verbose else 0)
        _set_threads_from_slurm(m)
        if timelimit is not None:
            m.setParam("TimeLimit", float(timelimit))

        w = {i: m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS) for i in groups}

        v = {(r, s): m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
             for r in items for s in items if r < s}

        m.setObjective(gp.quicksum(v[(r, s)] for r in items for s in items if r < s), GRB.MINIMIZE)

        # Weights must sum to 1
        m.addConstr(gp.quicksum(w[i] for i in groups) == 1)
        
        # Symmetry breaking: force weights to be ordered descendingly
        for i in range(1, g):
            m.addConstr(w[i] >= w[i + 1])

        # Linearized absolute value constraints
        for r in items:
            for s in items:
                if r < s:
                    cap = gp.quicksum(w[i] * xi.get((r, s, i), 0) for i in groups)
                    m.addConstr(v[(r, s)] >= c[r][s] - cap)
                    m.addConstr(v[(r, s)] >= cap - c[r][s])

        # Warm start
        if wi_start is not None:
            for i in groups:
                w[i].Start = float(wi_start[i])

            for r in items:
                for s in items:
                    if r < s:
                        cap0 = sum(wi_start[i] * xi.get((r, s, i), 0) for i in groups)
                        v[(r, s)].Start = abs(c[r][s] - cap0)

        m.optimize()

        if m.SolCount == 0:
            return None, None

        wi = [0.0] * (g + 1)
        for i in groups:
            wi[i] = w[i].X
        return wi, m.ObjVal

    finally:
        m.dispose()


# ============================================================
# Initializers (Multi-start mechanism)
# ============================================================
def make_wi_starts(g: int, k: int = 5, seed: int = 123) -> List[List[float]]:
    """Generates 'k' random, valid normalized weight vectors for multi-start."""
    rnd = random.Random(seed)
    starts: List[List[float]] = []

    for _ in range(k):
        vals = [rnd.random() + 1e-9 for _ in range(g)]
        vals.sort(reverse=True) # Symmetry breaking compliance
        s = sum(vals)
        
        wi = [0.0] * (g + 1)
        for i in range(1, g + 1):
            wi[i] = vals[i - 1] / s
        starts.append(wi)

    return starts

def xi_from_random_permutations(n: int, g: int, seed: int = 123):
    """Generates entirely random, transitive pairwise permutations for warm starting."""
    rnd = random.Random(seed)
    items = list(range(1, n + 1))
    xi = {}

    for i in range(1, g + 1):
        perm = items[:]
        rnd.shuffle(perm)
        pos = {perm[k]: k for k in range(n)}

        for r in items:
            for s in items:
                if r < s:
                    xi[(r, s, i)] = 1 if pos[r] < pos[s] else 0

    return xi


# ============================================================
# Output Generators
# ============================================================
def append_summary(summaryfile: Path, instance_path: Path,
                   n: int, g: int, nswaps: Optional[int],
                   it_completed: int, obj_last, obj_best, runtime: float,
                   best_run: int, starts_used: int):
    """Appends a single-line execution summary to a centralized log file."""
    summaryfile = summaryfile.resolve()
    summaryfile.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"{'instance':<30}"
        f"{'n':>6}"
        f"{'g':>6}"
        f"{'nswaps':>10}"
        f"{'it':>8}"
        f"{'obj_last':>18}"
        f"{'obj_best':>18}"
        f"{'runtime':>12}"
        f"{'run':>6}"
        f"{'starts':>8}\n"
    )

    if not summaryfile.exists():
        with summaryfile.open("w", encoding="utf-8") as f:
            f.write(header)

    line = (
        f"{instance_path.name:<30}"
        f"{n:>6d}"
        f"{g:>6d}"
        f"{'' if nswaps is None else nswaps:>10}"
        f"{it_completed:>8d}"
        f"{0.0 if obj_last is None else obj_last:>18.3f}"
        f"{0.0 if obj_best is None else obj_best:>18.3f}"
        f"{runtime:>12.3f}"
        f"{best_run:>6d}"
        f"{starts_used:>8d}\n"
    )

    with summaryfile.open("a", encoding="utf-8") as f:
        f.write(line)

def fmt_real(x, max_decimals=10):
    """Formats floats to strip trailing zeros for cleaner output."""
    s = f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s

def write_solution_like_example(outfile: Path, datafile: Path,
                                n: int, g: int,
                                xi: Dict[Tuple[int, int, int], int],
                                wi: List[float],
                                y: Dict[Tuple[int, int, int], float],
                                obj_like: int):
    """Writes the full, human-readable solution to the specified text file."""
    outfile = outfile.resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with outfile.open("a", encoding="utf-8") as f:
        f.write(f"{datafile}\n")
        f.write(f"n: {n}\n")
        f.write(f"g: {g}\n\n")

        total_group_obj = 0.0

        for i in range(1, g + 1):
            ranking = extract_ranking_from_xi(n, g, xi, i)

            f.write(f"GROUP PERMUTATION {i}\n")
            f.write("Ranking\n")
            f.write(" ".join(str(v) for v in ranking) + " \n")
            f.write("Weight\n")
            f.write(f"{wi[i]:.10g}\n")
            f.write("Matrix\n")

            for rr in range(1, n + 1):
                row_vals = []
                for ss in range(1, n + 1):
                    if rr == ss:
                        val = 0
                    else:
                        val_float = PRINT_SCALE * y[(rr, ss, i)]
                        val = int(round(val_float)) if PRINT_INT else val_float
                    row_vals.append(f"{val:4d}" if PRINT_INT else f"{val:8.3f}")
                f.write(" ".join(row_vals) + " \n")

            group_value = sum(
                pref_xi(xi, r, s, i) * (y[(r, s, i)] / wi[i])
                for r in range(1, n + 1) for s in range(1, n + 1)
                if r != s
            ) if wi[i] > 1e-12 else 0.0

            group_obj = (n * (n - 1)) / 2 - group_value
            total_group_obj += wi[i] * group_obj

            f.write(f"Objective value: {fmt_real(group_obj, 5)}\n\n\n")

        f.write(f"Total objective value: {fmt_real(total_group_obj, 5)}\n\n")


# ============================================================
# MULTI-START HEURISTIC PIPELINE
# ============================================================
def solve_hlop_heuristic_multistart(datafile: Path, outfile: Path, summaryfile: Path, g: int,
                                    restarts: int,
                                    it_tot: int,
                                    time_limit_total: float,
                                    verbose_phases: int = 0,
                                    seed: int = 123):
    """
    Main orchestration function. Executes the Alternating Minimization heuristic 
    multiple times (multi-start) from different randomized seeds to escape local optima.
    Tracks time strictly and halts when limits or convergence criteria are reached.
    """
    n, nswaps, a = load_instance(datafile)
    c = build_costs(a, n)

    env = gp.Env(empty=True)
    env.setParam("WLSTokenDuration", 60)
    env.setParam("WLSTokenRefresh", 1)
    env.start()

    try:
        wi_starts = make_wi_starts(g, k=restarts, seed=seed)

        t_global = time.time()
        starts_used = 0
        total_iters_completed = 0

        best_obj = None
        best_xi = None
        best_wi = None
        best_run = 0
        best_last_obj = None

        items = range(1, n + 1)
        groups = range(1, g + 1)

        for run_idx, wi0 in enumerate(wi_starts, start=1):
            remaining = time_limit_total - (time.time() - t_global)
            if remaining <= 1e-9:
                break

            starts_used += 1

            xi = xi_from_random_permutations(n, g, seed=seed + 1000 * run_idx)
            wi = list(wi0)

            # Evaluate initial solution to guarantee a feasible baseline
            initial_obj = evaluate_solution(n, g, c, xi, wi)

            local_best_obj = initial_obj
            local_best_xi = dict(xi)
            local_best_wi = list(wi)

            last_obj = initial_obj
            last_it_completed = 0

            nIt = 0
            while nIt < it_tot and (time.time() - t_global) <= time_limit_total:
                remaining_inner = time_limit_total - (time.time() - t_global)
                if remaining_inner <= 1e-9:
                    break

                # Save objective at the start of the loop for convergence checks
                obj_start_iteration = local_best_obj

                # --- PHASE 1 ---
                # Strict 120s limit per MIP phase to prevent stalling
                fase1_time = min(remaining_inner, 120.0)

                xi_new, obj1 = phase_1(
                    n, g, c, wi,
                    xi_start=xi,
                    verbose=verbose_phases,
                    timelimit=fase1_time,
                    env=env,
                )
                
                # Immediately update local best if improved
                if xi_new is not None and obj1 is not None:
                    if local_best_obj is None or obj1 < local_best_obj - 1e-12:
                        local_best_obj = obj1
                        local_best_xi = xi_new
                        local_best_wi = wi
                        last_obj = obj1
                
                if xi_new is None:
                    break

                xi = xi_new

                remaining_inner = time_limit_total - (time.time() - t_global)
                if remaining_inner <= 1e-9:
                    break

                # --- PHASE 2 ---
                wi_new, obj2 = phase_2(
                    n, g, c, xi,
                    wi_start=wi,
                    verbose=verbose_phases,
                    timelimit=remaining_inner,
                    env=env,
                )

                # Immediately update local best if improved
                if wi_new is not None and obj2 is not None:
                    if local_best_obj is None or obj2 < local_best_obj - 1e-12:
                        local_best_obj = obj2
                        local_best_xi = xi
                        local_best_wi = wi_new
                        last_obj = obj2
                
                if wi_new is None:
                    break

                wi = wi_new
                last_it_completed = nIt + 1
                total_iters_completed += 1

                # Convergence Criterion: Stop iterating if improvement is negligible
                if obj_start_iteration is not None and abs(obj_start_iteration - local_best_obj) < 1e-5:
                    break

                nIt += 1

            # GLOBAL MINIMIZATION: Compare best of this run vs overall best
            if local_best_obj is not None and (best_obj is None or local_best_obj < best_obj - 1e-12):
                best_obj = local_best_obj
                best_xi = local_best_xi
                best_wi = local_best_wi
                best_run = run_idx
                best_last_obj = last_obj

        total_runtime = time.time() - t_global

        if best_obj is None:
            # Fallback if solver fails completely
            print("No solution")
            append_summary(summaryfile, datafile, n, g, nswaps, total_iters_completed, None, None, total_runtime, 0, starts_used)
            return

        # Reconstruct optimal continuous variables y purely from the analytical Lemma
        y = build_y_min_structure(n, g, c, best_xi, best_wi)

        # Calculate final robust objective matching the initial MLOP formula
        v_sum = 0.0
        for r in items:
            for s in items:
                if r == s:
                    continue
                cap = sum(best_wi[i] * pref_xi(best_xi, r, s, i) for i in groups)
                v_sum += min(c[r][s], cap)
        obj_like = int(round(v_sum / 2.0))

        append_summary(summaryfile, datafile, n, g, nswaps, total_iters_completed, best_last_obj, best_obj, total_runtime, best_run, starts_used)

        print("Writing BEST solution to:", outfile.resolve())
        write_solution_like_example(outfile, datafile, n, g, best_xi, best_wi, y, obj_like)

    finally:
        env.dispose()


# ============================================================
# MAIN (CLI Entrypoint)
# ============================================================
def main():
    """Parses command-line arguments and triggers the optimization pipeline."""
    parser = argparse.ArgumentParser()
    
    # Made optional (nargs="?") with a default value to allow easy PyCharm execution
    parser.add_argument("datafile", nargs="?", default=str(DEFAULT_DATAFILE), help="Path to the .dat (passed by enqueue)")
    
    parser.add_argument("--g", type=int, default=DEFAULT_NUM_GRUPOS)
    parser.add_argument("--timelimit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--it_tot", type=int, default=DEFAULT_IT_TOT)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--outfile", default=str(DEFAULT_OUTFILE))
    parser.add_argument("--summaryfile", default=str(DEFAULT_SUMMARYFILE))
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    datafile = Path(args.datafile)

    print("SCRIPT:", Path(__file__).resolve())
    print("CWD:", Path.cwd())
    print("Reading instance:", datafile.resolve())
    print("SLURM_JOB_ID =", os.environ.get("SLURM_JOB_ID"))
    print("SLURM_JOB_NODELIST =", os.environ.get("SLURM_JOB_NODELIST"))
    print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))

    if not datafile.exists():
        raise FileNotFoundError(f"Cannot find the .dat at: {datafile.resolve()}")

    solve_hlop_heuristic_multistart(
        datafile=datafile,
        outfile=Path(args.outfile),
        summaryfile=Path(args.summaryfile),
        g=args.g,
        restarts=args.restarts,
        it_tot=args.it_tot,
        time_limit_total=args.timelimit,
        verbose_phases=0 if args.quiet else 1,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
