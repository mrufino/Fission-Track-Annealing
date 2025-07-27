#model_fit.py
import numpy as np
import pandas as pd
import json
from scipy.optimize import curve_fit

R_GAS = 1.987204258e-3  # kcal/(mol*K)

# --- Kinetic model functions ---
def f_PA(tT, c0, c1, c2):
    t, T = tT
    return c0 + c1 * np.log(t) + c2 / (R_GAS * T)

def f_PC(tT, c0, c1, c2):
    t, T = tT
    return c0 + c1 * np.log(t) + c2 * np.log(1 / (R_GAS * T))

def f_FA(tT, c0, c1, c2, c3):
    t, T = tT
    return c0 + c1 * (np.log(t) - c2) / (1 / (R_GAS * T) - c3)

def f_FC(tT, c0, c1, c2, c3):
    t, T = tT
    return c0 + c1 * (np.log(t) - c2) / (np.log(1 / (R_GAS * T)) - c3)

# --- Model fitting function ---
def DAM_fit(model_func, data, initial_params):
    """
    Fit the kinetic annealing model to data using nonlinear least squares.
    """
    t_data = data['tempoS'].values
    T_data = data['temperK'].values
    rho_data = data['rho'].values
    sigma_f = data['e.rho'].values

    # Model expects rho = 1 - exp(f_model), so define wrapper:
    def rho_model(tT, *params):
        f_val = model_func(tT, *params)
        f_val = np.clip(f_val, -700, 700) 
        return 1 - np.exp(f_val)

    # Perform curve fitting
    params, covariance = curve_fit(
        rho_model,
        (t_data, T_data),
        rho_data,
        sigma=sigma_f,
        absolute_sigma=True,
        p0=initial_params
    )

    errors = np.sqrt(np.diag(covariance))
    fitted_values = rho_model((t_data, T_data), *params)
    residuals = rho_data - fitted_values

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((rho_data - np.mean(rho_data))**2)
    r_squared = 1 - ss_res / ss_tot
    chi_squared = np.sum((residuals / sigma_f)**2)
    reduced_chi_squared = chi_squared / (len(rho_data) - len(params))

    return {
        'params': params,
        'errors': errors,
        'r_squared': r_squared,
        'chi_squared': chi_squared,
        'reduced_chi_squared': reduced_chi_squared,
        'fitted_values': fitted_values,
        'residuals': residuals,
        'sigma': sigma_f
    }

def fit_all_models(data):
    models = {
        "PA": (f_PA, [5.631, 0.1865, -10.46]),
        "PC": (f_PC, [-4.910, 0.1944, -9.610]),
        "FA": (f_FA, [-8.518, 0.1266, -20.99, 0.2985]),
        "FC": (f_FC, [-9.449, 0.1627, -24.58, -0.8626]),
    }
    results = []

    for model_name, (model_func, initial_params) in models.items():
        fit_result = DAM_fit(model_func, data, initial_params)
        res = {
            'model': model_name,
            'params': json.dumps(fit_result['params'].tolist()),
            'errors': json.dumps(fit_result['errors'].tolist()),
            'r_squared': fit_result['r_squared'],
            'chi_squared': fit_result['chi_squared'],
            'reduced_chi_squared': fit_result['reduced_chi_squared'],
            'residuals': json.dumps(fit_result['residuals'].tolist()),
            'fitted_values': json.dumps(fit_result['fitted_values'].tolist()),
            'sigma': json.dumps(fit_result['sigma'].tolist())
        }
        results.append(res)

    return pd.DataFrame(results)

def main():
    pass

if __name__ == "__main__":
    main()
