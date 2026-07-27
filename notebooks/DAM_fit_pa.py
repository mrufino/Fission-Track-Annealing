import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


R_GAS = 1.987204258e-3


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


def f_PA(tT, c0, c1, c2):
    t, T = tT
    return c0 + c1 * np.log(t) + c2 / (R_GAS * T)


def load_and_process_data(file_path):
    df = pd.read_csv(file_path, sep=r"\s+", engine="python")
    df = truncate_df_to_uncertainty(df, "r", "e.r")
    return df.iloc[:, [4, 5, 6, 7]].copy()


def fit_pa(data_path):
    data = load_and_process_data(data_path)
    t_data = data["tempoS"]
    T_data = data["temperK"]
    target_data = data["r"]
    sigma_f = data["e.r"]

    def target_model(tT, *params):
        return 1 - np.exp(f_PA(tT, *params))

    params, covariance = curve_fit(
        target_model,
        (t_data, T_data),
        target_data,
        sigma=sigma_f,
        absolute_sigma=True,
        p0=[5.631, 0.1865, -10.46],
    )

    errors = np.sqrt(np.diag(covariance))
    fitted_values = target_model((t_data, T_data), *params)
    residuals = target_data - fitted_values

    chi_squared = np.sum((residuals / sigma_f) ** 2)
    reduced_chi_squared = chi_squared / (len(target_data) - len(params))

    return {
        "params": params,
        "errors": errors,
        "covariance": covariance,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "sigma": sigma_f,
        "chi_squared": chi_squared,
        "reduced_chi_squared": reduced_chi_squared,
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "DurangoDataR.txt"
    results = fit_pa(data_path)

    out_path = Path(__file__).with_name("DAM_fit_pa_results.json")
    payload = {
        "params": results["params"].tolist(),
        "errors": results["errors"].tolist(),
        "covariance": results["covariance"].tolist(),
        "chi_squared": float(results["chi_squared"]),
        "reduced_chi_squared": float(results["reduced_chi_squared"]),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {out_path}")
    print("params (c0,c1,c2) =", payload["params"])
    print("errors =", payload["errors"])
    print("reduced_chi_squared =", payload["reduced_chi_squared"])
    print("covariance =")
    print(np.array(payload["covariance"]))


if __name__ == "__main__":
    main()
