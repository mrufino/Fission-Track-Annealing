import pandas as pd
import numpy as np

# Constants used in the conversion formula
K_CONST = 0.108
L0 = 14.24
N_POWER = 8
L0_UNCERTAINTY = 0.08

def convert_r_to_rho(r):
    """
    Convert track length ratio 'r' to normalized density 'rho'
    using empirical calibration for Durango apatite.

    Parameters:
    r : array-like
        Track length ratio.

    Returns:
    rho : array-like
        Normalized density corresponding to r.
    """
    numerator = r * (1 - 1 / (1 + (K_CONST * L0 * r) ** N_POWER) ** 2)
    denominator = 1 - 1 / (1 + (K_CONST * L0) ** N_POWER) ** 2
    return numerator / denominator

def propagate_uncertainty_rho(r, uncertainty_r):
    """
    Propagate uncertainties from r and L0 to uncertainty in rho.

    Parameters:
    r : array-like
        Track length ratio.
    uncertainty_r : array-like
        Uncertainty in r.

    Returns:
    uncertainty_rho : array-like
        Propagated uncertainty in rho.
    """
    k_l0 = K_CONST * L0
    k_l0_r = k_l0 * r

    term_k_l0_n = (1 + k_l0 ** N_POWER)
    term_k_l0r_n = (1 + k_l0_r ** N_POWER)

    # Partial derivatives terms for uncertainty propagation
    term_ul0 = (
        (2 * K_CONST * N_POWER * r**2 * k_l0_r**(N_POWER - 1)) / 
        ((1 - 1 / term_k_l0_n**2) * term_k_l0r_n**3)
        - (2 * K_CONST * k_l0**(N_POWER - 1) * N_POWER * r * (1 - 1 / term_k_l0r_n**2)) / 
        ((1 + k_l0**N_POWER)**3 * (1 - 1 / term_k_l0_n**2)**2)
    )

    term_ur = (
        (2 * k_l0 * N_POWER * r * k_l0_r**(N_POWER - 1)) / 
        ((1 - 1 / term_k_l0_n**2) * term_k_l0r_n**3)
        + (1 - 1 / term_k_l0r_n**2) / (1 - 1 / term_k_l0_n**2)
    )

    variance_l0 = (term_ul0 ** 2) * (L0_UNCERTAINTY ** 2)
    variance_r = (term_ur ** 2) * (uncertainty_r ** 2)

    return np.sqrt(variance_l0 + variance_r)

def truncate_to_uncertainty(value, uncertainty):
    """
    Round value to the first significant digit of its uncertainty.

    Parameters:
    value : float
        Measured value.
    uncertainty : float
        Associated uncertainty.

    Returns:
    rounded_value : float
        Rounded value respecting uncertainty significant digit.
    """
    if uncertainty == 0:
        return value
    decimal_places = -int(np.floor(np.log10(abs(uncertainty))))
    rounding_factor = 10 ** decimal_places
    return np.round(value * rounding_factor) / rounding_factor

def truncate_dataframe_values(df, value_col, uncertainty_col):
    """
    Apply rounding to a dataframe column's values based on their uncertainties.

    Parameters:
    df : pandas.DataFrame
        DataFrame with data.
    value_col : str
        Column name of values to round.
    uncertainty_col : str
        Column name of uncertainties.

    Returns:
    df : pandas.DataFrame
        DataFrame with truncated values and uncertainties.
    """
    df[value_col] = df.apply(
        lambda row: truncate_to_uncertainty(row[value_col], row[uncertainty_col]), axis=1
    )
    df[uncertainty_col] = df[uncertainty_col].apply(
        lambda x: truncate_to_uncertainty(x, x)
    )
    return df

def load_and_process_data(file_path):
    """
    Load raw data, truncate uncertainties, convert ratios to normalized density,
    propagate uncertainties, then truncate results.

    Parameters:
    file_path : str
        Path to raw data file.

    Returns:
    pandas.DataFrame
        Processed data with time, temperature, rho and uncertainty columns.
    """
    # Load data (assuming whitespace-separated file)
    df = pd.read_csv(file_path, sep=r"\s+", engine='python')

    # Round input 'r' and its uncertainty 'e.r' to first significant digit
    df = truncate_dataframe_values(df, 'r', 'e.r')

    # Prepare processed DataFrame with relevant columns copied
    processed_df = df.iloc[:, [4, 5, 6, 7]].copy()  # typically columns: time, temperature, r, e.r

    # Calculate normalized density rho from r
    processed_df.loc[:, 'rho'] = convert_r_to_rho(df['r'])

    # Propagate uncertainties into rho
    processed_df.loc[:, 'e.rho'] = propagate_uncertainty_rho(df['r'], df['e.r'])

    # Round rho and its uncertainty
    processed_df = truncate_dataframe_values(processed_df, 'rho', 'e.rho')

    return processed_df

if __name__ == "__main__":
    # Example usage: load and process data file
    processed = load_and_process_data("DurangoDataR.txt")
    print(processed.head())
