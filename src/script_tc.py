#!/usr/bin/env python3
"""
FINAL VERSION - MINIMAL CHANGES + REQUIRED MODIFICATIONS
"""

from __future__ import annotations

import argparse
import time
import pathlib
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import quad
from scipy.interpolate import interp1d
import multiprocessing

# --- 1. Constants and Symbolic Definitions ---
R_GAS_VAL = 1.987204258e-3  # Universal gas constant in kcal/(mol*K)
T_FINAL_K = 293.15  # Final temperature in Kelvin (20 ºC)
SECS_PER_MA = 3.1536e13  # Seconds per million years

# Symbolic variables for analytical formulation
u, T, R = sp.symbols('u T R')
a0, a1, a2, a3 = sp.symbols('a0 a1 a2 a3')

# Model equations depending on type (PA, PC, FA, FC)
MODEL_EXPRESSIONS = {
    'PA': a0 + a1 * sp.log(u) + a2 / (R * T),
    'PC': a0 + a1 * sp.log(u) + a2 * sp.log(1 / (R * T)),
    'FA': a0 + a1 * (sp.log(u) - a2) / (1 / (R * T) - a3),
    'FC': a0 + a1 * (sp.log(u) - a2) / (sp.log(1 / (R * T)) - a3)
}

def get_n_value(model_label: str, c1_val: float) -> float:
    """
    Compute the reaction order n based on the model type and c1 coefficient.
    """
    return (c1_val - 1.0) / c1_val if model_label in {"PA", "PC"} else 0.5

def calculate_tc_and_profiles(
    params: Dict, model_label: str, *,
    cooling_rate: float, t_max_ma: float, grid_N: int
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Main computation function for Tc, delta and gamma profiles.

    Returns:
        Tc_C (float): closure temperature in Celsius
        duration_points (np.ndarray): time vector (in seconds)
        delta_profile (np.ndarray): delta(t) evolution
        gamma_profile (np.ndarray): gamma(t) profile (integrated delta)
    """
    cooling_rate_s = cooling_rate / SECS_PER_MA
    t_max_s = t_max_ma * SECS_PER_MA

    # Substitute model coefficients
    subs_dict = {a0: params['c0'], a1: params['c1'], a2: params['c2'], R: R_GAS_VAL}
    if model_label in {"FA", "FC"}:
        subs_dict[a3] = params['c3']

    # Symbolic and numeric expressions
    model_numeric_expr = MODEL_EXPRESSIONS[model_label].subs(subs_dict)
    df_expr = sp.diff(model_numeric_expr, u)
    f_l = sp.lambdify((u, T), model_numeric_expr, 'numpy')
    df_l = sp.lambdify((u, T), df_expr, 'numpy')

    n = get_n_value(model_label, params['c1'])

    def k_rate(u_val, t_duration):
        T_val = T_FINAL_K + cooling_rate_s * (t_duration - u_val)
        try:
            lnk = np.log(np.abs(df_l(u_val, T_val))) - (n - 1) * f_l(u_val, T_val)
            return np.exp(lnk)
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0

    # Memoized delta(t)
    memo_delta = {}
    def delta(t_duration):
        if t_duration in memo_delta:
            return memo_delta[t_duration]
        if t_duration <= 1e-10:
            return 1.0

        integral, _ = quad(lambda u_val: k_rate(u_val, t_duration), 1e-10, t_duration, limit=150, epsrel=1e-4)
        base = (1 - n) * integral

        val = 1.0 - np.power(base, 1.0 / (1 - n)) if base >= 0 and np.isfinite(base) else 1.0
        result = np.clip(val, 0.0, 1.0)
        memo_delta[t_duration] = result
        return result

    duration_points = np.linspace(0, t_max_s, grid_N)
    delta_profile = np.array([delta(t) for t in duration_points])

    time_before_present = t_max_s - duration_points
    interp_func = interp1d(time_before_present, delta_profile, kind='linear', fill_value="extrapolate")
    gamma_val, _ = quad(interp_func, 0, t_max_s, limit=150, epsrel=1e-4)
    t_apparent_s = gamma_val

    Tc_K = T_FINAL_K + cooling_rate_s * t_apparent_s
    Tc_C = Tc_K - 273.15

    du_s = duration_points[1] - duration_points[0]
    gamma_profile = np.cumsum(delta_profile[::-1])[::-1] * du_s

    return Tc_C, duration_points, delta_profile, gamma_profile

def monte_carlo_worker(args: Tuple) -> float:
    """
    Monte Carlo sampling function for uncertainty propagation.
    Each worker samples one Tc based on normal-distributed parameters.
    """
    label, base_params, cooling_rate, t_max_ma, grid_N = args
    sampled_params = {
        key: np.random.normal(val, base_params.get(f"{key}_unc", 0.0))
        for key, val in base_params.items() if "_unc" not in key
    }
    try:
        tc_val, _, _, _ = calculate_tc_and_profiles(
            sampled_params, label,
            cooling_rate=cooling_rate, t_max_ma=t_max_ma, grid_N=grid_N
        )
        return tc_val
    except Exception:
        return np.nan

def load_fit_params_from_csv(csv_path: pathlib.Path) -> Dict[str, Dict]:
    """
    Load model parameters and uncertainties from a CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        exit(1)

    out = {}
    for _, row in df.iterrows():
        model = row["Model"].strip().upper()
        params = {}
        for col, val in row.items():
            if col == "Model": continue
            clean_key = col.strip().replace('_', '').replace('err', '_unc')
            if pd.isna(val): val = 0.0
            params[clean_key] = float(val)
        out[model] = params
    return out

def main():
    """
    Main entry point for CLI execution. Handles argument parsing,
    Monte Carlo loop, and CSV export for results and profiles.
    """
    parser = argparse.ArgumentParser(description="Final Tc calculator with SymPy.")
    parser.add_argument("--csv", required=True, type=pathlib.Path, help="CSV file with model parameters.")
    parser.add_argument("--rate", type=float, default=1.0, help="Cooling rate (K/Ma).")
    parser.add_argument("--n_iter", type=int, default=1000, help="Monte Carlo iterations.")
    parser.add_argument("--N", type=int, default=100, help="Number of time grid points.")
    args = parser.parse_args()

    cooling_rate = args.rate
    t_max_ma = 250.0 / cooling_rate  # Automatically adjusted based on rate

    params_dict = load_fit_params_from_csv(args.csv)

    rate_str = str(args.rate).replace('.', 'p')
    output_dir = os.path.join("TC_results_csv", f"{rate_str}KMa")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nResults (Tc ± SD) for cooling rate = {args.rate:.3f} K/Ma:\n")

    summary_data = []

    for label in ("PA", "PC", "FA", "FC"):
        if label not in params_dict: continue

        start_time = time.time()
        print(f"Calculating Tc for model {label}...")

        mc_args = [(label, params_dict[label], cooling_rate, t_max_ma, args.N)] * args.n_iter

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(monte_carlo_worker, mc_args)

        valid_results = [r for r in results if np.isfinite(r)]
        if len(valid_results) < args.n_iter * 0.9:
            print(f"  \033[93mWarning: {args.n_iter - len(valid_results)} out of {args.n_iter} iterations failed.\033[0m")

        mean_tc = np.mean(valid_results)
        sd_tc = np.std(valid_results, ddof=1) if len(valid_results) > 1 else 0.0
        se_tc = sd_tc / np.sqrt(len(valid_results)) if len(valid_results) > 1 else 0.0

        summary_data.append({
            'Model': label,
            'Tc': mean_tc,
            'SD': sd_tc,
            'SE': se_tc
        })

        duration = time.time() - start_time
        print(f"  Monte Carlo completed in {duration:.2f} seconds.")
        print(f"  \033[92m{label}: Tc = {mean_tc:7.2f} °C   ± {sd_tc:.2f} (SD)\033[0m")

        print("  Computing and saving mean profiles...")
        mean_params = {key: val for key, val in params_dict[label].items() if "_unc" not in key}
        _, duration_points, delta_profile, gamma_profile = calculate_tc_and_profiles(
            mean_params, label,
            cooling_rate=cooling_rate, t_max_ma=t_max_ma, grid_N=args.N
        )

        time_ma = duration_points / SECS_PER_MA
        temp_start_C = (T_FINAL_K + (cooling_rate / SECS_PER_MA) * duration_points) - 273.15

        df_delta = pd.DataFrame({'Time_Ma': time_ma, 'Initial_Temperature_C': temp_start_C, 'Mean_Delta': delta_profile})
        df_gamma = pd.DataFrame({
            'Time_Before_Present_Ma': t_max_ma - time_ma,
            'Temperature_C': temp_start_C,
            'Mean_Gamma_Ma': gamma_profile / SECS_PER_MA
        })

        df_delta.to_csv(os.path.join(output_dir, f"delta_profile_{label}_{rate_str}KMa.csv"), index=False, float_format='%.4f')
        df_gamma.to_csv(os.path.join(output_dir, f"gamma_profile_{label}_{rate_str}KMa.csv"), index=False, float_format='%.4f')
        print(f"  Profiles saved for {label}.")

    # Export final Tc summary
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, f"tc_summary_{rate_str}KMa.csv"), index=False, float_format="%.4f")
    print(f"\nSummary saved to tc_summary_{rate_str}KMa.csv.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
