"""
Lévy Stable Distribution Fitting for S&P500 Returns
Fits a Lévy stable distribution to S&P500 log returns and compares it with normal distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import pickle

# MAGIC NUMBERS
HISTORICAL_PERIOD = "10y" # Period of data retrieval
OUTPUT_PICKLE_FILE = 'levy_ergebnisse.pkl'
N_POINTS_PDF = 1000 # Number of points used to compute PDF
N_BINS_MAIN = 100 # Number of bins for histogram
N_BINS_TAIL = 50 # Number of bins for tail histogram
OUTPUT_FIGURE_FILE = 'levy_distribution'
DPI_SETTINGS = 300

# Function for Lévy parameter fitting on S&P500 data
def parameter_bestimmen(log_rendite):

    params = stats.levy_stable.fit(log_rendite)
    alpha, beta, loc, scale = params

    print(f"Alpha (Tail-Index): {alpha:.6f}")
    print(f"Beta (Symmetry): {beta:.6f}")
    print(f"Mü (Mean): {loc:.4f}")
    print(f"Gamma (Scattering): {scale:.6f}")

    return alpha, beta, loc, scale

def main():

    # Fetch S&P500 historical data
    ticker = yf.Ticker("^GSPC")
    data = ticker.history(period = HISTORICAL_PERIOD)

    # Calculate log returns
    log_rendite = np.log(data['Close'] / data['Close'].shift(1))
    log_rendite = log_rendite.dropna()

    # Calculate descriptive statitics of S&P500 data
    mittelwert = log_rendite.mean()
    std_abweichung = log_rendite.std()
    schiefe = stats.skew(log_rendite)  
    woelbung = stats.kurtosis(log_rendite)

    # Print Statistics
    print(f"Datapoints: {len(log_rendite)}")
    print(f"Mean: {log_rendite.mean():.6f}")
    print(f"Standard Deviation: {log_rendite.std():.6f}")
    print(f"Kurtosis: {stats.kurtosis(log_rendite):.4f}")

    alpha, beta, loc, scale = parameter_bestimmen(log_rendite)

    # Save statistics and fitted values for further analysis
    ergebnisse = {
        'log_rendite': log_rendite.values,  
        'alpha': alpha,
        'beta': beta,
        'loc': loc,
        'scale': scale,
        'mittelwert': mittelwert,
        'std_abweichung': std_abweichung,
        'schiefe': schiefe,
        'woelbung': woelbung
    }

    with open(OUTPUT_PICKLE_FILE, 'wb') as f:
        pickle.dump(ergebnisse, f)

    # Create visualization with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: PDF comparison (Normal vs Lévy)
    ax1 = axes[0, 0]
    ax1.hist(log_rendite, bins=N_BINS_MAIN, density=True, alpha=0.7, color='blue', label='Empirical Data', edgecolor='black')
    x= np.linspace(start =min(log_rendite), stop= max(log_rendite), num =N_POINTS_PDF, endpoint = True)

    # Fitted Normal distribution PDF
    normal_fit = stats.norm.pdf(x, loc = log_rendite.mean(), scale=log_rendite.std())
    ax1.plot(x, normal_fit, 'r-', linewidth=2, label = 'Normal Distribution')

    # Fitted Lévy distribution PDF
    levy_fit = stats.levy_stable.pdf(x, alpha, beta, loc, scale)
    ax1.plot(x, levy_fit, 'g-', linewidth=2, label=f'Lévy (α={alpha:.2f})')

    ax1.set_xlabel('Return')
    ax1.set_ylabel('Density')
    ax1.set_title('Returns-distribution: Normal vs. Lévy')
    ax1.legend()
    ax1.grid(True, alpha = 0.3)

    # Subplot 2: Heavy tail visualization through log scale 
    ax2 = axes[0, 1]
    x_pos = x[x > 0]

    norm_pos = stats.norm.pdf(x_pos, loc = log_rendite.mean(), scale = log_rendite.std())
    levy_pos = stats.levy_stable.pdf(x_pos, alpha, beta, loc, scale)

    ax2.semilogy(x_pos, norm_pos, 'r-', linewidth=2, label='Normal (Exponential)')
    ax2.semilogy(x_pos, levy_pos, 'g-', linewidth=2, label='Lévy (Power-law)')

    ax2.hist(log_rendite[log_rendite > 0], bins=N_BINS_TAIL, density=True, alpha=0.5, color='blue', label='Data (pos.)')

    ax2.set_xlabel('Return')
    ax2.set_ylabel('log(Density)')
    ax2.set_title('Right Tail (Log scale)')
    ax2.legend()
    ax2.grid(True, alpha = 0.3)

    # Subplot 3: QQ-Plot for quantile cross-check for the normal distribution
    ax3 = axes[1, 0]

    stats.probplot(log_rendite, dist="norm", plot=ax3)

    ax3.set_title('QQ-Plot: Data vs. Normal Distribution')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Cumulative Distribution Function comparison
    ax4 = axes[1, 1]

    # Sort data for CDF representation
    sort_log_rendite = np.sort(log_rendite)

    # Empirical CDF
    empirisches_cdf = np.arange(1, len(sort_log_rendite) + 1)/ len(sort_log_rendite)
    ax4.plot(sort_log_rendite, empirisches_cdf, 'b-', linewidth=1.5, label='Empirical CDF', alpha=0.7)

    # Normal CDF
    normal_cdf = stats.norm.cdf(sort_log_rendite, loc=log_rendite.mean(), scale=log_rendite.std())

    # Lévy CDF
    levy_cdf = stats.levy_stable.cdf(sort_log_rendite, alpha, beta, loc, scale)

    ax4.plot(sort_log_rendite, normal_cdf, 'r--', linewidth=2, label='Normal CDF')
    ax4.plot(sort_log_rendite, levy_cdf, 'g--', linewidth=2, label='Lévy CDF')


    ax4.set_xlabel('Return')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Density Function')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Display plots
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_FILE, dpi=DPI_SETTINGS)
    plt.show()

if __name__ == "__main__":
    main()