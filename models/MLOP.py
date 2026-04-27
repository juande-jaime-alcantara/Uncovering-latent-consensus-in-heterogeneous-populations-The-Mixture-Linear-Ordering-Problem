#!/usr/bin/env python3
"""
Mixture Linear Ordering Problem (MLOP) Solver.

This script uses Gurobi to solve the MLOP by parsing a custom `.dat` instance,
formulating the Mixed-Integer Linear Programming (MILP) model, and outputting both 
a detailed solution report and a summary of the execution.
"""

import re
import os
import argparse
import time
import random
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

# ============================================================
# Global Configuration & Default Parameters
# ============================================================

# Base directory of the script. Used as an anchor to safely resolve
# relative paths regardless of where the script is executed from.
BASE_DIR = Path(__file__).resolve().parent

# --- File Paths ---

# Default instance to solve if no datafile is provided via command line.
DEFAULT_DATAFILE = BASE_DIR / "../inputs/sushi.dat"

# Path to store the detailed, human-readable solution report (rankings, matrices, etc.).
DEFAULT_OUTFILE = BASE_DIR / "../outputs/solutionsMLOP.txt"

# Path to append a one-line performance summary of the execution.
# Highly useful for tracking results across batch experiments.
DEFAULT_SUMMARYFILE = BASE_DIR / "../outputs/summaryMLOP.txt"

# --- Optimization Parameters ---

# Default number of heterogeneous groups (g) to partition the population into.
DEFAULT_NUM_GRUPOS = 2

# Maximum Gurobi solver runtime in seconds (14400s = 4 hours).
DEFAULT_TIME_LIMIT = 14400

# --- Output Formatting ---

# Controls how the reconstructed Y variables are printed in the output file.
# To see the strictly normalized continuous variables [0, 1] as defined in
# the mathematical model (Lemma 2), use PRINT_SCALE = 1 and PRINT_INT = False.
#
# PRINT_SCALE: Multiplier for the y-values (e.g., set to 1000 to scale up fractions).
# PRINT_INT: If True, rounds the scaled y-values to integers for cleaner matrices.
PRINT_SCALE = 1
PRINT_INT = False


# ============================================================
# .dat parser
# ============================================================

def _parse_scalar_int(text: str, key: str) -> Optional[int]:
    """
    Extracts an integer value associated with a specific key from the text.
    """
    m = re.search(rf"\b{re.escape(key)}\s*:\s*(-?\d+)\b", text)
    return int(m.group(1)) if m else None


def _parse_matrix_a(text: str, n: int) -> List[List[int]]:
    """
    Extracts the matrix 'a' from the text file. 
    Assumes a 1-based indexing structure of size (n+1)x(n+1) for convenience.
    """
    m = re.search(r"\ba\s*:\s*\[\s*(.*?)\s*\]", text, flags=re.DOTALL | re.MULTILINE)
    if not m:
        raise ValueError("Cannot find block 'a: [ ... ]'")

    nums = re.findall(r"-?\d+", m.group(1))
    vals = [int(x) for x in nums]

    if len(vals) != n * n:
        raise ValueError("Incorrect size for matrix a")

    a = [[0] * (n + 1) for _ in range(n + 1)]
    k = 0
    for r in range(1, n + 1):
        for s in range(1, n + 1):
            a[r][s] = vals[k]
            k += 1
    return a


def load_instance(path: Path) -> Tuple[int, Optional[int], List[List[int]]]:
    """
    Loads an instance from a given file path.
    Returns the number of items (n), number of swaps (nswaps), and the frequency matrix (a).
    """
    txt = path.read_text(encoding="utf-8", errors="ignore")
    n = _parse_scalar_int(txt, "n")
    if n is None:
        raise ValueError("Cannot find n in the instance file")
    nswaps = _parse_scalar_int(txt, "nswaps")
    a = _parse_matrix_a(txt, n)
    return n, nswaps, a


# ============================================================
# Rounding function
# ============================================================

def fmt_real(x: float, max_decimals: int = 10) -> str:
    """
    Formats a float to a string, stripping trailing zeros to keep the output clean.
    """
    s = f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


# ============================================================
# Summary Logger
# ============================================================

def append_summary(summaryfile: Path, instance_path: Path, n: int, g: int,
                   nswaps: Optional[int], model: gp.Model) -> None:
    """
    Appends a one-line summary of the Gurobi optimization run to a summary file.
    Includes objective values, bounds, gap, nodes explored, and runtime.
    """
    summaryfile = summaryfile.resolve()
    summaryfile.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"{'instance':<30}{'n':>6}{'g':>6}{'nswaps':>10}"
        f"{'obj':>12}{'bound':>12}{'gap%':>10}"
        f"{'sols':>8}{'nodes':>12}{'status':>10}{'runtime':>12}\n"
    )

    if not summaryfile.exists():
        with summaryfile.open("w", encoding="utf-8") as f:
            f.write(header)

    if model.SolCount > 0:
        obj = model.ObjVal
        gap = getattr(model, "MIPGap", float("nan"))
    else:
        obj = float("nan")
        gap = float("nan")

    bound = getattr(model, "ObjBound", float("nan"))
    solcount = getattr(model, "SolCount", 0)
    nodecount = getattr(model, "NodeCount", 0.0)

    line = (
        f"{instance_path.name:<30}{n:>6d}{g:>6d}"
        f"{'' if nswaps is None else nswaps:>10}"
        f"{obj:>12.3f}{bound:>12.3f}{(gap*100):>10.3f}"
        f"{solcount:>8d}{nodecount:>12.0f}{model.Status:>10d}{model.Runtime:>12.3f}\n"
    )

    with summaryfile.open("a", encoding="utf-8") as f:
        f.write(line)


# ============================================================
# Helper: Extracting preferences
# ============================================================

def pref_x(x: Dict[Tuple[int, int, int], gp.Var], r: int, s: int, i: int) -> float:
    """
    Retrieves the value (relaxed or integer) indicating if item r is preferred over s in group i.
    Since the decision variables x are only defined for r < s to reduce symmetry/model size, 
    this helper dynamically infers the reverse relationship (1 - x) when r > s.
    """
    if r == s:
        return 0.0
    if r < s:
        return float(x[(r, s, i)].X)
    else:
        return 1.0 - float(x[(s, r, i)].X)


# ============================================================
# POSTPROCESSING: Ranking and Reconstruction
# ============================================================

def extract_ranking(items: range, x: Dict, i: int) -> Tuple[List[int], Dict[int, float]]:
    """
    Generates a ranking of items for a specific group 'i' from best to worst.
    The score of an item 'r' is determined by how many items 's' it defeats (sum of preferences).
    """
    scores: Dict[int, float] = {}
    for r in items:
        scores[r] = sum(pref_x(x, r, s, i) for s in items if s != r)
    ranking = sorted(items, key=lambda r: scores[r], reverse=True)
    return ranking, scores


def reconstruct_y_from_x_w(items: range, groups: range, x: Dict, w: Dict, c: List[List[float]], eps: float = 1e-12) -> Dict[Tuple[int, int, int], float]:
    """
    Reconstructs the continuous variables y_{rs}^i from the binary layout x and weights w.
    Utilizes the min-structure lemma to allocate probability mass appropriately 
    across the groups prioritizing the actual preferences.
    """
    omega = {i: w[i].X for i in groups}
    Y: Dict[Tuple[int, int, int], float] = {}

    for r in items:
        for s in items:
            if r >= s:
                continue

            crs = c[r][s]
            csr = c[s][r]

            # Partition groups based on preference r ≻ s
            A = [i for i in groups if pref_x(x, r, s, i) > 0.5]
            B = [i for i in groups if i not in A]

            WA = sum(omega[i] for i in A)
            WB = sum(omega[i] for i in B)

            if crs <= WA + eps:
                if WA <= eps:
                    for i in groups:
                        Y[(i, r, s)] = 0.0
                        Y[(i, s, r)] = omega[i]
                else:
                    alpha = crs / WA
                    for i in groups:
                        y_rs = alpha * omega[i] if i in A else 0.0
                        Y[(i, r, s)] = y_rs
                        Y[(i, s, r)] = omega[i] - y_rs
            else:
                if WB <= eps:
                    for i in groups:
                        y_rs = omega[i] if i in A else 0.0
                        Y[(i, r, s)] = y_rs
                        Y[(i, s, r)] = omega[i] - y_rs
                else:
                    alpha = csr / WB
                    for i in groups:
                        y_sr = alpha * omega[i] if i in B else 0.0
                        y_rs = omega[i] - y_sr
                        Y[(i, r, s)] = y_rs
                        Y[(i, s, r)] = y_sr

    return Y


# ============================================================
# Output Formatting
# ============================================================

def write_solution_like_example(outfile: Path, datafile: Path, n: int, g: int, model: gp.Model,
                                items: range, groups: range, x: Dict, w: Dict, c: List[List[float]]) -> None:
    """
    Writes the detailed breakdown of the model's solution to an output file.
    Includes the ranking, weights, and reconstructed Y matrix for each group.
    """
    outfile = outfile.resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with outfile.open("a", encoding="utf-8") as f:
        f.write(f"{datafile}\n")
        f.write(f"n: {n}\n")
        f.write(f"g: {g}\n\n")

        if model.SolCount == 0:
            f.write("No solution found.\n\n")
            return

        Y = reconstruct_y_from_x_w(items, groups, x, w, c)

        for i in groups:
            ranking, _ = extract_ranking(items, x, i)
            wi = w[i].X

            f.write(f"GROUP {i}\n")
            f.write("Ranking\n")
            f.write(" ".join(str(v) for v in ranking) + " \n")
            f.write("Weight\n")
            f.write(f"{wi:.10g}\n")
            f.write("Matrix\n")

            for rr in items:
                row_vals = []
                for ss in items:
                    if rr == ss:
                        val = 0
                    else:
                        y = Y[(i, rr, ss)]
                        val = int(round(PRINT_SCALE * y)) if PRINT_INT else (PRINT_SCALE * y)
                    row_vals.append(f"{val:4d}" if PRINT_INT else f"{val:8.3f}")
                f.write(" ".join(row_vals) + " \n")

            group_value = sum(
                pref_x(x, r, s, i) * (Y[(i, r, s)] / wi)
                for r in items for s in items
                if r != s
            ) if wi > 1e-12 else 0.0

            group_obj = n * (n - 1) / 2 - group_value

            f.write(f"Objective value: {fmt_real(group_obj, 5)}\n\n\n")

        f.write(f"Total objective value: {fmt_real(model.ObjVal, 5)}\n\n")


# ============================================================
# Gurobi WLS Environment Wrapper
# ============================================================

def make_wls_env(max_retries: int = 3) -> gp.Env:
    """
    Creates and starts a Gurobi Web License Service (WLS) environment robustly.
    Implements a retry mechanism with exponential backoff to handle intermittent 
    'Token expired' errors during environment initialization.
    """
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        env = gp.Env(empty=True)
        env.setParam("WLSTokenDuration", 60)
        env.setParam("WLSTokenRefresh", 1)

        try:
            env.start()
            return env
        except gp.GurobiError as e:
            last_err = e
            try:
                env.dispose()
            except Exception:
                pass

            # Backoff before retrying
            sleep_s = 2.0 * attempt + random.random()
            time.sleep(sleep_s)

    raise last_err if last_err is not None else RuntimeError("Could not start the WLS Env")


# ============================================================
# Core MIP Solver
# ============================================================

def solve_mlop(datafile: Path, outfile: Path, summaryfile: Path, g: int,
               time_limit: Optional[float], verbose: int) -> None:
    """
    Formulates and solves the Mixed-Integer Programming model for the MLOP.
    """
    n, nswaps, a = load_instance(datafile)

    items = range(1, n + 1)
    groups = range(1, g + 1)

    # Precompute relative frequencies c_{rs} = a_{rs} / (a_{rs} + a_{sr})
    # If the denominator is 0 (no observations), default to 0.5 (ties)
    c = [[0] * (n + 1) for _ in range(n + 1)]
    for r in items:
        for s in items:
            if r == s:
                continue
            denom = a[r][s] + a[s][r]
            c[r][s] = 0.5 if denom == 0 else a[r][s] / denom

    env = make_wls_env(max_retries=3)
    model: gp.Model | None = None

    try:
        model = gp.Model("MLOP", env=env)

        # Solver parameters
        model.setParam("MIPGap", 1e-5)
        model.setParam("OutputFlag", 1 if verbose else 0)
        model.setParam("LogToConsole", 1)

        # Automatically allocate threads if running within a Slurm environment
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus and slurm_cpus.isdigit():
            model.setParam("Threads", int(slurm_cpus))

        if time_limit:
            model.setParam("TimeLimit", float(time_limit))

        # --------------------------------------------------------
        # Decision Variables
        # --------------------------------------------------------
        
        # x_{rs}^i: 1 if item r precedes item s in group i. (Defined only for r < s)
        x = {(r, s, i): model.addVar(vtype=GRB.BINARY, name=f"x_{r}_{s}_{i}")
             for r in items for s in items for i in groups if r < s}

        # w_{i}: Continuous weight assigned to group i.
        w = {i: model.addVar(lb=0, ub=1, name=f"w_{i}") for i in groups}

        # u_{rs}^i: Continuous linearization variable representing w_i * x_{rs}^i
        u = {(r, s, i): model.addVar(lb=0, ub=1, name=f"u_{r}_{s}_{i}")
             for r in items for s in items for i in groups if r < s}

        # v_{rs}: Absolute error variable between the target frequency and the model's prediction
        v = {(r, s): model.addVar(lb=0, name=f"v_{r}_{s}")
             for r in items for s in items if r < s}

        model.update()

        # --------------------------------------------------------
        # Objective Function
        # --------------------------------------------------------
        # Minimize the sum of absolute errors across all strictly upper-triangular pairs
        model.setObjective(
            gp.quicksum(v[(r, s)] for r in items for s in items if r < s),
            GRB.MINIMIZE
        )

        # --------------------------------------------------------
        # Constraints
        # --------------------------------------------------------
        
        # 1. Transitivity within each group (3-cycle inequalities)
        # For any triplet (r, s, t) where r < s < t, ensure strict linear ordering.
        for i in groups:
            for r in items:
                for s in items:
                    if s <= r:
                        continue
                    for t in items:
                        if t <= s:
                            continue
                        model.addConstr(x[(r, s, i)] + x[(s, t, i)] - x[(r, t, i)] >= 0)
                        model.addConstr(x[(r, s, i)] + x[(s, t, i)] - x[(r, t, i)] <= 1)

        # 2. Weight normalization and symmetry breaking
        model.addConstr(gp.quicksum(w[i] for i in groups) == 1)
        for i in range(1, g):
            model.addConstr(w[i] >= w[i + 1])  # Order groups by weight to break symmetry

        # 3. McCormick Envelope Linearization: u_{rs}^i = w_i * x_{rs}^i
        # Since x is binary, this exact linearization requires 3 constraints.
        for i in groups:
            for r in items:
                for s in items:
                    if r < s:
                        model.addConstr(u[(r, s, i)] <= w[i])
                        model.addConstr(u[(r, s, i)] <= x[(r, s, i)])
                        model.addConstr(u[(r, s, i)] >= w[i] + x[(r, s, i)] - 1)

        # 4. Absolute value constraints for v_{rs} = | c_{rs} - sum_i u_{rs}^i |
        for r in items:
            for s in items:
                if r < s:
                    sum_u = gp.quicksum(u[(r, s, i)] for i in groups)
                    model.addConstr(v[(r, s)] >= c[r][s] - sum_u)
                    model.addConstr(v[(r, s)] >= sum_u - c[r][s])

        # Execute optimization
        model.optimize()

        # --------------------------------------------------------
        # Post-Processing
        # --------------------------------------------------------
        append_summary(summaryfile, datafile, n, g, nswaps, model)

        print("STATUS =", model.Status, "SOLCOUNT =", model.SolCount)
        print("Writing solution to:", outfile.resolve())

        write_solution_like_example(
            outfile=outfile,
            datafile=datafile,
            n=n,
            g=g,
            model=model,
            items=items,
            groups=groups,
            x=x,
            w=w,
            c=c,
        )

    finally:
        # Resource cleanup
        try:
            if model is not None:
                model.dispose()
        finally:
            env.dispose()


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    """
    Parses command-line arguments and triggers the optimization pipeline.
    """
    parser = argparse.ArgumentParser(description="Solve the MLOP optimization model via Gurobi.")

    parser.add_argument("datafile", nargs="?", default=str(DEFAULT_DATAFILE), help="Path to the instance .dat file")
    parser.add_argument("--g", type=int, default=DEFAULT_NUM_GRUPOS, help="Number of groups (default: 4)")
    parser.add_argument("--timelimit", type=float, default=DEFAULT_TIME_LIMIT, help="Time limit in seconds")
    parser.add_argument("--outfile", default=str(DEFAULT_OUTFILE), help="Path for the detailed solution output")
    parser.add_argument("--summaryfile", default=str(DEFAULT_SUMMARYFILE), help="Path for the single-line summary log")
    parser.add_argument("--quiet", action="store_true", help="Suppress Gurobi console output")

    args = parser.parse_args()

    datafile = Path(args.datafile)

    print("SCRIPT:", Path(__file__).resolve())
    print("CWD:", Path.cwd())
    print("Reading instance:", datafile.resolve())
    print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))

    solve_hlop(
        datafile=datafile,
        outfile=Path(args.outfile),
        summaryfile=Path(args.summaryfile),
        g=args.g,
        time_limit=args.timelimit,
        verbose=0 if args.quiet else 1,
    )


if __name__ == "__main__":
    main()
