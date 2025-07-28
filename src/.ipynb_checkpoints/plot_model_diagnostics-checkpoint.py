#plot_model_diagnostics.py
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_diagnostics(results_df, model_name):
    row = results_df[results_df['model'] == model_name].iloc[0]

    residuals = np.array(json.loads(row['residuals']))
    sigma = np.array(json.loads(row['sigma']))
    fitted = np.array(json.loads(row['fitted_values']))
    norm_residuals = residuals / sigma

    print(f"Model: {model_name}")
    print(f"Mean residual: {np.mean(residuals):.4f}")
    print(f"Std residual: {np.std(residuals):.4f}")
    print(f"Outliers (>3σ): {np.sum(np.abs(norm_residuals) > 3)}")

    # Plot 1: Resíduos vs Fitted
    plt.figure(figsize=(8,5))
    plt.errorbar(fitted, residuals, yerr=sigma, fmt='o', color='red', ecolor='black', capsize=3, alpha=0.7)
    plt.axhline(0, linestyle='--', color='black')
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Fitted - {model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Histograma dos resíduos normalizados
    plt.figure(figsize=(8,4))
    plt.hist(norm_residuals, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Normalized Residuals Histogram - {model_name}")
    plt.xlabel("Normalized Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Plot 3: Q-Q plot
    plt.figure(figsize=(6,6))
    stats.probplot(norm_residuals, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot - {model_name}")
    plt.tight_layout()
    plt.show()
