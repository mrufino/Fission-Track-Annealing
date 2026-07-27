import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


R_GAS = 1.987204258e-3


def convert_r_rho(r):
    k = 0.108
    l0 = 14.24
    n = 8
    numerator = r * (1 - 1 / (1 + (k * l0 * r) ** n) ** 2)
    denominator = 1 - 1 / (1 + (k * l0) ** n) ** 2
    return numerator / denominator


def convert_rho_to_r(rho):
    r_grid = np.linspace(0.0, 1.0, 20001)
    rho_grid = convert_r_rho(r_grid)
    rho_values = np.asarray(rho, dtype=float)
    rho_clipped = np.clip(rho_values, rho_grid.min(), rho_grid.max())
    return np.interp(rho_clipped, rho_grid, r_grid)


def truncate_to_first_significant(value, uncertainty):
    if uncertainty == 0:
        return value
    decimal_places = -int(np.floor(np.log10(abs(uncertainty))))
    factor = 10 ** decimal_places
    return np.round(value * factor) / factor


def truncate_df_to_uncertainty(df, target_column, uncertainty_column):
    df = df.copy()
    df[target_column] = df.apply(lambda row: truncate_to_first_significant(row[target_column], row[uncertainty_column]), axis=1)
    df[uncertainty_column] = df[uncertainty_column].apply(lambda x: truncate_to_first_significant(x, x))
    return df


def f_FC(tT, c0, c1, c2, c3):
    t, T = tT
    return c0 + c1 * (np.log(t) - c2) / (np.log(1 / (R_GAS * T)) - c3)


def load_and_process_data(file_path):
    df = pd.read_csv(file_path, sep=r"\s+", engine="python")
    df = truncate_df_to_uncertainty(df, "r", "e.r")
    return df.iloc[:, [4, 5, 6, 7]].copy()


def fit_fc(data_path):
    data = load_and_process_data(data_path)
    t_data = data["tempoS"]
    T_data = data["temperK"]
    target_data = data["r"]
    sigma_f = data["e.r"]

    def target_model(tT, *params):
        predicted_rho = 1 - np.exp(f_FC(tT, *params))
        return convert_rho_to_r(predicted_rho)

    params, covariance = curve_fit(
        target_model,
        (t_data, T_data),
        target_data,
        sigma=sigma_f,
        absolute_sigma=True,
        p0=[-9.449, 0.1627, -24.58, -0.8626],
    )

    errors = np.sqrt(np.diag(covariance))
    fitted_values = target_model((t_data, T_data), *params)
    residuals = target_data - fitted_values

    return {
        "params": params,
        "errors": errors,
        "covariance": covariance,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "sigma": sigma_f,
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "DurangoDataR.txt"
    results = fit_fc(data_path)

    out_path = Path(__file__).with_name("DAM_fit_fc_results.json")
    payload = {
        "params": results["params"].tolist(),
        "errors": results["errors"].tolist(),
        "covariance": results["covariance"].tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {out_path}")
    print("params =", payload["params"])
    print("covariance =")
    print(np.array(payload["covariance"]))


if __name__ == "__main__":
    main()