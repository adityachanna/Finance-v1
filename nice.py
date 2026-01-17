import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import ScalarFormatter
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import gamma, nbinom
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')

# ========================
# CONSTANTS - Named magic numbers for clarity
# ========================
TRADING_DAYS_WEEK = 5
TRADING_DAYS_MONTH = 21
TRADING_DAYS_QUARTER = 63
TRADING_DAYS_HALF_YEAR = 126
TRADING_DAYS_YEAR = 252
MA_SHORT = 50
MA_LONG = 200
RSI_WINDOW = 14
BB_WINDOW = 20
VOL_WINDOW = 21
CORR_WINDOW = 63
DRAWDOWN_WINDOW = 126

# ========================
# HSMM Wrapper - Extends GaussianHMM with Duration Modeling
# ========================

class GaussianHSMM:
    """
    Hidden Semi-Markov Model built on top of hmmlearn's GaussianHMM.

    Key additions:
    1. Duration distributions for each state (Gamma or Negative Binomial)
    2. Duration-aware decoding
    3. Post-processing to enforce duration constraints
    """

    def __init__(self, n_components=3, covariance_type="diag",
                 duration_type="gamma", min_duration=5, max_duration=100,
                 n_iter=100, random_state=42, verbose=False):
        """
        Parameters:
        -----------
        n_components : int
            Number of hidden states
        covariance_type : str
            'diag' or 'full' covariance for emissions
        duration_type : str
            'gamma' or 'nbinom' for duration distribution
        min_duration : int
            Minimum duration in a state (prevents rapid switching)
        max_duration : int
            Maximum duration for computational efficiency
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.duration_type = duration_type
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

        # Initialize base HMM
        self.hmm = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            verbose=verbose
        )

        # Duration parameters (will be estimated)
        self.duration_params_ = None

    def fit(self, X):
        """
        Fit HSMM in two stages:
        1. Fit base HMM to learn emission/transition structure
        2. Estimate duration distributions from decoded states
        """
        if self.verbose:
            print("Stage 1: Fitting base HMM...")

        # Fit standard HMM
        self.hmm.fit(X)

        if self.verbose:
            print("Stage 2: Estimating duration distributions...")

        # Decode states
        states = self.hmm.predict(X)

        # Estimate duration parameters for each state
        self.duration_params_ = {}

        for k in range(self.n_components):
            durations = self._extract_durations(states, k)

            if len(durations) > 0:
                if self.duration_type == "gamma":
                    # Fit Gamma distribution
                    mean_d = np.mean(durations)
                    var_d = np.var(durations)

                    # Method of moments
                    scale = var_d / mean_d if mean_d > 0 else 1.0
                    shape = mean_d / scale if scale > 0 else 1.0

                    self.duration_params_[k] = {
                        'shape': max(shape, 0.5),  # Ensure valid shape
                        'scale': max(scale, 1.0),
                        'mean': mean_d,
                        'median': np.median(durations)
                    }

                elif self.duration_type == "nbinom":
                    # Fit Negative Binomial
                    mean_d = np.mean(durations)
                    var_d = np.var(durations)

                    # Method of moments
                    p = mean_d / var_d if var_d > mean_d else 0.5
                    n = mean_d * p / (1 - p) if p < 1 else 1

                    self.duration_params_[k] = {
                        'n': max(n, 1),
                        'p': min(max(p, 0.01), 0.99),
                        'mean': mean_d,
                        'median': np.median(durations)
                    }
            else:
                # Default parameters if no observations
                self.duration_params_[k] = {
                    'shape': 2.0, 'scale': 10.0, 'mean': 20.0, 'median': 20.0
                } if self.duration_type == "gamma" else {
                    'n': 5, 'p': 0.2, 'mean': 20.0, 'median': 20.0
                }

        if self.verbose:
            print("\nDuration Statistics by State:")
            for k in range(self.n_components):
                params = self.duration_params_[k]
                print(f"  State {k}: mean={params['mean']:.1f}, median={params['median']:.1f}")

        return self

    def _extract_durations(self, states, state_id):
        """Extract all durations for a given state"""
        durations = []
        current_duration = 0

        for t in range(len(states)):
            if states[t] == state_id:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return durations

    def predict(self, X):
        """
        Duration-aware state prediction.
        Uses base HMM then enforces minimum duration constraints.
        """
        # Get base HMM prediction
        states_raw = self.hmm.predict(X)

        # Post-process to enforce minimum duration
        states_smoothed = self._enforce_min_duration(states_raw)

        return states_smoothed

    def _enforce_min_duration(self, states):
        """Remove spurious short-duration states"""
        n = len(states)
        states_smooth = states.copy()

        i = 0
        while i < n:
            current_state = states_smooth[i]

            # Find run length
            j = i
            while j < n and states_smooth[j] == current_state:
                j += 1

            duration = j - i

            # If too short, assign to neighbor with higher probability
            # WARNING: This uses look-ahead (next_state) which causes bias in live prediction.
            # For real-time use, consider only using prev_state or disabling smoothing.
            if duration < self.min_duration and duration < n:

                # Look at neighbors
                prev_state = states_smooth[i-1] if i > 0 else None
                next_state = states_smooth[j] if j < n else None

                # Assign to most common neighbor
                if prev_state is not None and next_state is not None:
                    replace_state = prev_state if prev_state == next_state else prev_state
                elif prev_state is not None:
                    replace_state = prev_state
                elif next_state is not None:
                    replace_state = next_state
                else:
                    replace_state = current_state

                states_smooth[i:j] = replace_state

            i = j

        return states_smooth

    def score(self, X):
        """Compute log-likelihood using base HMM"""
        return self.hmm.score(X)

    def get_duration_stats(self):
        """Return duration statistics for all states"""
        return pd.DataFrame(self.duration_params_).T


def hsmm_bic(hsmm_model, X):
    """Calculate BIC for HSMM"""
    K = hsmm_model.n_components
    T, d = X.shape

    # Use base HMM log-likelihood
    logL = hsmm_model.score(X)

    # Parameters: HMM params + 2 duration params per state
    if hsmm_model.covariance_type == "diag":
        hmm_params = (K-1) + K*(K-1) + K*d + K*d
    else:
        hmm_params = (K-1) + K*(K-1) + K*d + K*d*(d+1)/2

    duration_params = 2 * K  # shape, scale for gamma
    k_params = hmm_params + duration_params

    bic = -2.0 * logL + k_params * np.log(T)

    return bic, logL


# ========================
# MAIN EXECUTION
# ========================
# Note: This script is designed to run as a standalone analysis script.
# The if __name__ check below ensures the code doesn't execute on import.

if __name__ != "__main__":
    raise SystemExit("This script is meant to be run directly, not imported.")

print("=" * 60)
print("HSMM MARKET REGIME DETECTION")
print("=" * 60)
print("\n--- Step 1: Data Collection ---")

ticker_map = {
    "^GSPC": "SPX",
    "^IXIC": "NDX",
    "^RUT": "RUT",
    "^VIX": "VIX"
}

tickers = list(ticker_map.keys())
print(f"Fetching data for: {list(ticker_map.values())}")

raw_data_full = yf.download(tickers, period="max", progress=False, auto_adjust=False)

if isinstance(raw_data_full.columns, pd.MultiIndex):
    try:
        raw_data = raw_data_full['Adj Close']
    except KeyError:
        raw_data = raw_data_full['Close']
else:
    raw_data = raw_data_full

raw_data = raw_data.rename(columns=ticker_map)
start_dates = raw_data.apply(lambda x: x.first_valid_index())
common_start_date = start_dates.max()

print(f"Common start date: {common_start_date.date()}")

cutoff_date = "2022-12-31"
df = raw_data[common_start_date : cutoff_date].copy()
df = df.ffill().dropna().reset_index()
if 'Date' not in df.columns:
    df = df.rename(columns={'index': 'Date'})

print(f"Training data: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Shape: {df.shape}")

# ========================
# 2. Feature Engineering
# ========================
print("\n--- Step 2: Feature Engineering ---")

def calc_rsi(series, window=RSI_WINDOW):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_bollinger_width(series, window=BB_WINDOW):
    """Calculate Bollinger Band width as (upper - lower) / ma = (4 * std) / ma"""
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + (2 * std)
    lower = ma - (2 * std)
    return (upper - lower) / ma  # Consistent formula: 4*std/ma

def realized_vol(series, window=VOL_WINDOW):
    return series.rolling(window).std() * np.sqrt(TRADING_DAYS_YEAR)

def calc_drawdown(series, window=DRAWDOWN_WINDOW):
    roll_max = series.rolling(window, min_periods=1).max()
    return series / roll_max - 1.0

# Price-based features
for col in ["SPX", "NDX", "RUT"]:
    # Multi-timeframe returns
    df[f"{col}_Ret_5D"] = df[col].pct_change(5)
    df[f"{col}_Ret_21D"] = df[col].pct_change(21)
    df[f"{col}_Ret_63D"] = df[col].pct_change(63)
    df[f"{col}_Ret_126D"] = df[col].pct_change(126)

    # Technical indicators
    df[f"{col}_RSI"] = calc_rsi(df[col])
    df[f"{col}_BB_Width"] = calc_bollinger_width(df[col])

    # Trend metrics
    df[f"{col}_Dist_MA50"] = (df[col] / df[col].rolling(50).mean()) - 1.0
    df[f"{col}_Dist_MA200"] = (df[col] / df[col].rolling(200).mean()) - 1.0

    # Daily returns for other calculations
    df[f"{col}_Daily_Ret"] = df[col].pct_change()

# Volatility metrics
df["SPX_RealVol"] = realized_vol(df["SPX_Daily_Ret"])
df["NDX_RealVol"] = realized_vol(df["NDX_Daily_Ret"])
df["SPX_Skew_63D"] = df["SPX_Daily_Ret"].rolling(63).skew()

# VIX dynamics
df["VIX_vs_MA50"] = df["VIX"] / df["VIX"].rolling(50).mean()
df["VIX_Change_5D"] = df["VIX"].diff(5)
df["Vol_Risk_Premium"] = df["VIX"] - (df["SPX_RealVol"] * 100)

# Cross-asset relationships
window_corr = 63
df["Corr_NDX_SPX"] = df["NDX_Daily_Ret"].rolling(window_corr).corr(df["SPX_Daily_Ret"])
df["Corr_RUT_SPX"] = df["RUT_Daily_Ret"].rolling(window_corr).corr(df["SPX_Daily_Ret"])

cov_rut = df["RUT_Daily_Ret"].rolling(window_corr).cov(df["SPX_Daily_Ret"])
var_spx = df["SPX_Daily_Ret"].rolling(window_corr).var()
df["Beta_RUT_SPX"] = cov_rut / var_spx

df["Ratio_NDX_SPX"] = df["NDX"] / df["SPX"]
df["Ratio_NDX_SPX_Trend"] = df["Ratio_NDX_SPX"].pct_change(63)

# Drawdowns
df["SPX_Drawdown"] = calc_drawdown(df["SPX"])
df["NDX_Drawdown"] = calc_drawdown(df["NDX"])

# Clean
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
print(f"After feature engineering: {df_clean.shape}")
print("\n--- Step 3: Feature Selection & Dimensionality Reduction ---")

# Define feature groups for better interpretability
feature_groups = {
    'momentum': [
        # SPX momentum (all timeframes)
        'SPX_Ret_5D', 'SPX_Ret_21D', 'SPX_Ret_63D', 'SPX_Ret_126D',
        # NDX momentum (all timeframes) - was missing 5D and 126D
        'NDX_Ret_5D', 'NDX_Ret_21D', 'NDX_Ret_63D', 'NDX_Ret_126D',
        # RUT momentum - WAS CALCULATED BUT NEVER USED (small cap divergence signals)
        'RUT_Ret_5D', 'RUT_Ret_21D', 'RUT_Ret_63D', 'RUT_Ret_126D'
    ],
    'trend': [
        'SPX_Dist_MA50', 'SPX_Dist_MA200',
        'NDX_Dist_MA50', 'NDX_Dist_MA200',
        # RUT trend - WAS CALCULATED BUT NEVER USED
        'RUT_Dist_MA50', 'RUT_Dist_MA200'
    ],
    'technical': [
        'SPX_RSI', 'SPX_BB_Width',
        'NDX_RSI', 'NDX_BB_Width',
        # RUT technicals - WAS CALCULATED BUT NEVER USED
        'RUT_RSI', 'RUT_BB_Width'
    ],
    'volatility': [
        'SPX_RealVol', 'NDX_RealVol', 'SPX_Skew_63D',
        'VIX', 'VIX_vs_MA50', 'VIX_Change_5D', 'Vol_Risk_Premium'
    ],
    'structure': [
        'SPX_Drawdown', 'NDX_Drawdown',
        'Corr_NDX_SPX', 'Corr_RUT_SPX',
        'Beta_RUT_SPX', 'Ratio_NDX_SPX_Trend'
    ]
}

# Flatten all features
all_features = []
for group in feature_groups.values():
    all_features.extend(group)

print(f"Total features selected: {len(all_features)}")
print("\nFeature groups:")
for name, feats in feature_groups.items():
    print(f"  {name}: {len(feats)} features")

# Prepare feature matrix
X_df = df_clean[all_features].replace([np.inf, -np.inf], np.nan).dropna()
valid_idx = X_df.index

print(f"\nValid samples: {len(valid_idx)} ({len(valid_idx)/len(df_clean)*100:.1f}%)")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df.values)
print("\n--- PCA Analysis ---")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
evr = pca.explained_variance_ratio_

# Analyze variance explained
cumvar = np.cumsum(evr)
print("\nVariance explained by components:")
for i in [5, 10, 15, 20]:
    if i < len(cumvar):
        print(f"  First {i} PCs: {cumvar[i-1]:.1%}")

# Multiple selection strategies
n_pcs_95 = np.argmax(cumvar >= 0.95) + 1
n_pcs_90 = np.argmax(cumvar >= 0.90) + 1
n_pcs_elbow = np.argmax(evr < 0.05) + 1  # Where marginal gain drops below 5%

print(f"\nComponent selection:")
print(f"  90% variance: {n_pcs_90} components")
print(f"  95% variance: {n_pcs_95} components")
print(f"  Elbow method: {n_pcs_elbow} components")

# Choose middle ground - balance information and complexity
n_components = min(n_pcs_90, 15)  # Cap at 15 to avoid overfitting
X_final = X_pca[:, :n_components]

print(f"\n✓ Selected {n_components} components ({cumvar[n_components-1]:.1%} variance)")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(cumvar[:30], marker='o', linewidth=2)
ax1.axhline(y=0.90, color='r', linestyle='--', label='90%')
ax1.axhline(y=0.95, color='g', linestyle='--', label='95%')
ax1.axvline(x=n_components-1, color='b', linestyle='--', label=f'Selected ({n_components})')
ax1.set_xlabel('Number of Components')
ax1.set_ylabel('Cumulative Variance Explained')
ax1.set_title('PCA: Cumulative Variance')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar(range(1, min(31, len(evr)+1)), evr[:30], alpha=0.7)
ax2.axvline(x=n_components, color='r', linestyle='--', linewidth=2, label=f'Selected ({n_components})')
ax2.set_xlabel('Component')
ax2.set_ylabel('Variance Explained')
ax2.set_title('PCA: Individual Component Variance')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("\n--- Step 4: HSMM Model Selection ---")

state_range = range(3, 10)  # Test 3-7 states
results = []
best = {"bic": np.inf, "hsmm": None, "K": None, "logL": None}

for K in state_range:
    print(f"\nTesting K={K} states...")
    best_for_K = {"bic": np.inf, "hsmm": None, "logL": None}

    # Multiple random initializations for robustness
    for seed in range(5):
        hsmm = GaussianHSMM(
            n_components=K,
            covariance_type="diag",
            duration_type="gamma",
            min_duration=5,  # Prevent rapid switching
            max_duration=100,
            n_iter=100,
            random_state=42 + seed,
            verbose=False
        )

        hsmm.fit(X_final)
        bic, logL = hsmm_bic(hsmm, X_final)

        if bic < best_for_K["bic"]:
            best_for_K = {"bic": bic, "hsmm": hsmm, "logL": logL}

    results.append((K, best_for_K["bic"], best_for_K["logL"]))

    if best_for_K["bic"] < best["bic"]:
        best = {"bic": best_for_K["bic"], "hsmm": best_for_K["hsmm"],
                "K": K, "logL": best_for_K["logL"]}

    print(f"  Best BIC: {best_for_K['bic']:,.1f}, LogL: {best_for_K['logL']:,.1f}")

print(f"\n{'='*60}")
print(f"OPTIMAL MODEL: {best['K']} states (BIC={best['bic']:,.1f})")
print(f"{'='*60}")
print("\n--- Step 5: Regime Analysis ---")

hsmm_best = best["hsmm"]
final_states = hsmm_best.predict(X_final)

# Add regime to dataframe
df_clean["Regime"] = pd.NA
df_clean.loc[valid_idx, "Regime"] = final_states
df_clean["Regime"] = df_clean["Regime"].astype("Int64")

# Comprehensive state diagnostics
spx_ret = df_clean.loc[valid_idx, "SPX_Daily_Ret"].to_numpy()
spx_vol = df_clean.loc[valid_idx, "SPX_RealVol"].to_numpy()
spx_dd = df_clean.loc[valid_idx, "SPX_Drawdown"].to_numpy()
vix = df_clean.loc[valid_idx, "VIX"].to_numpy()

rows = []
for s in range(best["K"]):
    mask = (final_states == s)

    # Extract durations
    durations = hsmm_best._extract_durations(final_states, s)

    rows.append({
        "State": s,
        "Count": int(mask.sum()),
        "Pct_%": mask.sum() / len(final_states) * 100,
        "Avg_Duration": np.mean(durations) if len(durations) > 0 else 0,
        "Med_Duration": np.median(durations) if len(durations) > 0 else 0,
        "Daily_Ret_%": np.nanmean(spx_ret[mask]) * 100,
        "Ann_Ret_%": np.nanmean(spx_ret[mask]) * 252 * 100,
        "Ann_Vol_%": np.nanmean(spx_vol[mask]) * 100,
        "Avg_DD_%": np.nanmean(spx_dd[mask]) * 100,
        "Avg_VIX": np.nanmean(vix[mask])
    })

diag = pd.DataFrame(rows).sort_values("Ann_Ret_%", ascending=False).reset_index(drop=True)
def label_regime(row):
    """
    More nuanced regime labeling based on returns, volatility, and VIX.
    
    Classification logic:
    - Crisis: Deep losses (< -20%) with panic VIX (> 30)
    - Bear Market: Significant losses (< -15%) without full panic
    - Correction: Moderate losses (-5% to -15%)
    - High Volatility: Small returns but chaotic (VIX > 25)
    - Sideways/Range: Low returns, low volatility
    - Recovery: Positive returns with elevated VIX (post-crash bounce)
    - Bull Market: Strong returns with calm VIX
    - Quiet Bull: Moderate returns with very calm conditions
    """
    ret = row["Ann_Ret_%"]
    vix = row["Avg_VIX"]
    vol = row["Ann_Vol_%"]
    
    # Crisis: Extreme losses with panic
    if ret < -25 and vix > 30:
        return "Crisis"
    
    # Bear Market: Deep losses but not full panic
    if ret < -15:
        if vix > 25:
            return "Bear Market"
        else:
            return "Correction"
    
    # Correction: Moderate negative returns
    if ret < -5:
        return "Correction"
    
    # High Volatility sideways: Near-zero returns but chaotic
    if abs(ret) < 10 and (vix > 25 or vol > 20):
        return "High Volatility"
    
    # Sideways/Range: Low returns, calm market
    if abs(ret) < 8:
        return "Sideways"
    
    # Positive returns territory
    if ret > 0:
        # Recovery: Good returns but still elevated stress
        if vix > 20:
            return "Recovery"
        # Strong Bull: Excellent returns, calm market
        if ret > 20 and vix < 16:
            return "Strong Bull"
        # Bull Market: Good returns, reasonable VIX
        if ret > 12:
            return "Bull Market"
        # Mild Bull: Modest positive returns
        return "Mild Bull"
    
    return "Sideways"

diag["Label"] = diag.apply(label_regime, axis=1)

print("\n" + "="*80)
print("REGIME DIAGNOSTICS (Ranked by Returns)")
print("="*80)
print(diag.to_string(index=False))

# Duration statistics
print("\n" + "="*80)
print("DURATION PARAMETERS")
print("="*80)
duration_stats = hsmm_best.get_duration_stats()
print(duration_stats.to_string())

print("\n✓ HSMM Analysis Complete!")
print(f"  Model: {best['K']} states with explicit duration modeling")
print(f"  Training period: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")
print(f"  Features: {len(all_features)} → {n_components} PCs")

# ========================
# Visualization: Price Chart (Readable Axis + Fixed Colors)
# ========================

# 1. Filter Data (2000-2020)
mask = (df_clean["Date"] >= "2000-01-01") & (df_clean["Date"] <= "2020-12-31")
df_plot = df_clean.loc[mask].copy()

# 2. Logic to Fix Colors
regime_stats = df_plot.groupby("Regime")["VIX"].mean().sort_values()
sorted_regime_ids = regime_stats.index.tolist()
colors_ordered = sns.color_palette("RdYlGn_r", n_colors=len(sorted_regime_ids))

regime_color_map = {}
for rank, regime_id in enumerate(sorted_regime_ids):
    regime_color_map[regime_id] = colors_ordered[rank]

# 3. Prepare Plotting Data
x = mdates.date2num(df_plot["Date"])
y = df_plot["SPX"].values
regimes = df_plot["Regime"].values

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
segment_colors = [regime_color_map[r] for r in regimes[:-1]]

# 4. Plot
fig, ax = plt.subplots(figsize=(14, 7))

lc = LineCollection(segments, colors=segment_colors, linewidths=1.5)
ax.add_collection(lc)

# --- KEY FIX: Y-AXIS FORMATTING (The Price) ---
ax.set_yscale("log") # Keep log scale for accuracy
ax.set_ylim(y.min() * 0.9, y.max() * 1.1)
ax.set_xlim(x.min(), x.max())

# Force specific price labels so they appear clearly on the left
price_ticks = [700, 1000, 1500, 2000, 3000, 4000]
ax.set_yticks(price_ticks)
ax.get_yaxis().set_major_formatter(ScalarFormatter()) # Remove "10^3" notation
ax.ticklabel_format(style='plain', axis='y') # Force plain numbers (e.g. 1500)

# --- X-Axis Date Formatting ---
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=0)

# Add plot title and labels
ax.set_title('S&P 500 Price with Market Regimes (2000-2020)', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('S&P 500 Price (Log Scale)', fontsize=12)

# Create a legend
legend_handles = []
for regime_id in sorted_regime_ids:
    label = diag[diag['State'] == regime_id]['Label'].iloc[0] # Get label from diagnostic table
    color = regime_color_map[regime_id]
    patch = mpatches.Patch(color=color, label=label)
    legend_handles.append(patch)

ax.legend(handles=legend_handles, title='Market Regimes', bbox_to_anchor=(1.05, 1), loc='upper left')

ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
# ========================
# OUT-OF-SAMPLE TEST: 2023 to NOW
# ========================

# 1. Fetch Fresh Data (with buffer for Moving Averages)
# We start in mid-2022 so the 200-day MA is ready by Jan 1, 2023
# IMPORTANT: Include ^RUT here as some feature_cols depend on it
tickers = ["^GSPC", "^IXIC", "^RUT", "^VIX"] # Added ^RUT
new_data = yf.download(tickers, start="2022-06-01", progress=False, auto_adjust=False)

# Clean & Rename
if isinstance(new_data.columns, pd.MultiIndex):
    try:
        df_new = new_data['Adj Close'].copy()
    except KeyError:
        df_new = new_data['Close'].copy()
else:
    df_new = new_data.copy()

df_new = df_new.rename(columns=ticker_map).ffill().dropna().reset_index()
if 'Date' not in df_new.columns:
    df_new = df_new.rename(columns={'index': 'Date'})

# 2. Re-Calculate Features (Exact same formulas as before)
# We must use the EXACT same logic so the model understands the input
print("\n--- Re-engineering Features for Out-of-Sample Data ---")

# --- Helper Functions (re-defined or ensured accessible) ---
def calc_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_bollinger_width(series, window=20):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + (2 * std)
    lower = ma - (2 * std)
    return (upper - lower) / ma

def realized_vol(series, window=21):
    return series.rolling(window).std() * np.sqrt(252)

def calc_drawdown(series, window=126):
    roll_max = series.rolling(window, min_periods=1).max()
    return series / roll_max - 1.0

# --- A. Core Price Transformations ---
for col in ["SPX", "NDX", "RUT"]:
    # 1. Returns (Momentum Stack)
    df_new[f"{col}_Ret_5D"]   = df_new[col].pct_change(5)
    df_new[f"{col}_Ret_21D"]  = df_new[col].pct_change(21)
    df_new[f"{col}_Ret_63D"]  = df_new[col].pct_change(63)
    df_new[f"{col}_Ret_126D"] = df_new[col].pct_change(126)
    df_new[f"{col}_Ret_252D"] = df_new[col].pct_change(252)

    # 2. Technical Indicators
    df_new[f"{col}_RSI"] = calc_rsi(df_new[col])
    df_new[f"{col}_BB_Width"] = calc_bollinger_width(df_new[col])

    # 3. Trend Deviation (Distance from MAs)
    df_new[f"{col}_Dist_MA50"]  = (df_new[col] / df_new[col].rolling(50).mean()) - 1.0
    df_new[f"{col}_Dist_MA200"] = (df_new[col] / df_new[col].rolling(200).mean()) - 1.0

# --- B. Volatility & Risk Metrics ---
for col in ["SPX", "NDX", "RUT"]:
    df_new[f"{col}_Daily_Ret"] = df_new[col].pct_change()

df_new["SPX_RealVol"] = realized_vol(df_new["SPX_Daily_Ret"])
df_new["NDX_RealVol"] = realized_vol(df_new["NDX_Daily_Ret"])

df_new["SPX_Skew_63D"] = df_new["SPX_Daily_Ret"].rolling(63).skew()

df_new["VIX_vs_MA50"] = df_new["VIX"] / df_new["VIX"].rolling(50).mean()
df_new["VIX_Change_5D"] = df_new["VIX"].diff(5)

df_new["Vol_Risk_Premium"] = df_new["VIX"] - (df_new["SPX_RealVol"] * 100)

# --- C. Cross-Asset Dynamics ---
window_corr = 63

df_new["Corr_NDX_SPX"] = df_new["NDX_Daily_Ret"].rolling(window_corr).corr(df_new["SPX_Daily_Ret"])
df_new["Corr_RUT_SPX"] = df_new["RUT_Daily_Ret"].rolling(window_corr).corr(df_new["SPX_Daily_Ret"])

df_new["Ratio_NDX_SPX"] = df_new["NDX"] / df_new["SPX"]
df_new["Ratio_NDX_SPX_Trend"] = df_new["Ratio_NDX_SPX"].pct_change(63)

cov_rut = df_new["RUT_Daily_Ret"].rolling(window_corr).cov(df_new["SPX_Daily_Ret"])
var_spx = df_new["SPX_Daily_Ret"].rolling(window_corr).var()
df_new["Beta_RUT_SPX"] = cov_rut / var_spx

# --- D. Drawdowns ---
df_new["SPX_Drawdown"] = calc_drawdown(df_new["SPX"])
df_new["NDX_Drawdown"] = calc_drawdown(df_new["NDX"])

# 3. Filter for 2023 -> Now (and drop NaNs caused by rolling calculations)
df_recent = df_new[df_new["Date"] >= "2023-01-01"].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# 4. Transform Data (Use the EXISTING scaler/pca/model)
# We do NOT fit_transform here, only 'transform' to keep the model frozen
X_recent = df_recent[all_features].values # Changed feature_cols to all_features
X_recent_scaled = scaler.transform(X_recent) # Use old scaler
X_recent_pca = pca.transform(X_recent_scaled)[:, :n_components] # Apply n_components selection here

# Predict using the old HMM
recent_regimes = hsmm_best.predict(X_recent_pca) # Changed hmm_best to hsmm_best
df_recent["Regime"] = recent_regimes

# 5. Visualization (2023-Present)
# Use the SAME color map from the previous step for consistency
x = mdates.date2num(df_recent["Date"])
y = df_recent["SPX"].values
regimes = df_recent["Regime"].values

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
segment_colors = [regime_color_map[r] for r in regimes[:-1]]

fig, ax = plt.subplots(figsize=(14, 7))
lc = LineCollection(segments, colors=segment_colors, linewidths=2) # Thicker line for recent data
ax.add_collection(lc)

# Formatting
ax.set_yscale("log")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min() * 0.98, y.max() * 1.02)

# Specific Price Ticks for the recent range (4000 to 6000+)
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # Every 3 months
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")) # "Jan 2023"
plt.xticks(rotation=45)

# Add Legend
ax.legend(handles=legend_handles, loc="upper left", title="Market State")

plt.title(f"Out-of-Sample Test: S&P 500 Regimes (2023 - Present)", fontsize=14)
plt.ylabel("S&P 500 Price (Log Scale)", fontsize=12, fontweight='bold') # Changed label from '$' to 'Log Scale'
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print latest status
latest_regime = df_recent["Regime"].iloc[-1]
latest_vix = df_recent["VIX"].iloc[-1]
print(f"Current Market State (As of {df_recent['Date'].iloc[-1].date()}):\nRegime: {latest_regime} ({'Safe/Green' if latest_regime == sorted_regime_ids[0] else 'Risky/Red'})\nCurrent VIX: {latest_vix:.2f}")

# ==========================================
# 1. HSMM Helper: Extract Duration Distributions
# ==========================================
def get_hsmm_parameters(df, state_col='Regime'):
    """
    Analyzes historical Viterbi path to extract:
    1. Duration Distributions (How long do we usually stay in State X?)
    2. State Transition Matrix (Where do we go when we leave State X?)
    """
    states = df[state_col].values
    n_states = len(np.unique(states))
    
    # A. Calculate Durations
    # run_lengths[s] = list of all durations observed for state s
    run_lengths = {s: [] for s in range(n_states)}
    
    # transitions[from_state][to_state] = count
    transitions = np.zeros((n_states, n_states))
    
    current_state = states[0]
    current_run = 0
    
    for i in range(1, len(states)):
        # Increment duration
        current_run += 1
        
        next_state = states[i]
        
        if next_state != current_state:
            # Regime Switch occurred
            run_lengths[current_state].append(current_run)
            transitions[current_state, next_state] += 1
            
            # Reset
            current_state = next_state
            current_run = 0
            
    # Normalize Transitions to get probabilities (Conditional on SWITCHING)
    # i.e., Given I am leaving State 0, what is prob I go to 1 vs 2?
    row_sums = transitions.sum(axis=1, keepdims=True)
    
    # FIX: Handle zero-sum rows by assigning uniform probability to other states
    # This prevents division by zero when a state never transitions
    trans_probs = np.zeros_like(transitions)
    for i in range(transitions.shape[0]):
        if row_sums[i] > 0:
            trans_probs[i] = transitions[i] / row_sums[i]
        else:
            # Uniform distribution over OTHER states (not self)
            n_other = transitions.shape[0] - 1
            if n_other > 0:
                trans_probs[i] = 1.0 / n_other
                trans_probs[i, i] = 0  # No self-transition
            else:
                trans_probs[i] = 1.0  # Edge case: only 1 state
    
    return run_lengths, trans_probs

# ==========================================
# 2. The HSMM Monte Carlo Engine
# ==========================================
def hsmm_monte_carlo(durations_map, trans_probs, df_hist, 
                     start_state, start_price, 
                     days_ahead=30, n_sims=1000, random_state=None):
    """
    Monte Carlo simulation using HSMM duration properties.
    
    Parameters:
    -----------
    random_state : int, optional
        Seed for reproducibility. If None, results will vary between runs.
    """
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    future_paths = np.zeros((n_sims, days_ahead))
    
    # Pre-cache historical returns for bootstrapping
    state_returns = {}
    for s in durations_map.keys():
        rets = df_hist.loc[df_hist["Regime"] == s, "SPX_Daily_Ret"].dropna().values
        state_returns[s] = rets

    for sim_i in range(n_sims):
        
        current_price = start_price
        current_state = start_state
        
        # Track days simulated so far
        day_count = 0
        path = []
        
        while day_count < days_ahead:
            
            # A. Sample Duration (How long do we stay?)
            # We sample from historical observed durations for this state
            # (Adding a small random noise or jitter can help avoid overfitting specific values)
            possible_durations = durations_map[current_state]
            if len(possible_durations) > 0:
                stay_duration = np.random.choice(possible_durations)
            else:
                stay_duration = 5 # Fallback if history is empty
            
            # B. Simulate for that duration
            for t in range(stay_duration):
                if day_count >= days_ahead:
                    break
                
                # Sample Return from THIS state
                r = np.random.choice(state_returns[current_state])
                current_price = current_price * (1 + r)
                path.append(current_price)
                day_count += 1
            
            # C. Force Switch (Semi-Markov Step)
            # We must leave current_state. Use the transition matrix derived from switches only.
            next_state = np.random.choice(
                a=range(len(durations_map)),
                p=trans_probs[current_state]
            )
            current_state = next_state
            
        future_paths[sim_i, :] = path
        
    return future_paths

# ==========================================
# 3. Run Simulation
# ==========================================
print("--- Step 1: Extracting HSMM Parameters from History ---")

# Calculate Durations and Switch Probabilities from your 'df_clean'
# (Make sure df_clean has the 'Regime' column from the training step)
durations_map, hsmm_trans_mat = get_hsmm_parameters(df_clean)

# Show Duration Statistics
print("\nState Duration Statistics (Days):")
for s in durations_map:
    d = durations_map[s]
    print(f"  State {s}: Mean={np.mean(d):.1f} days, Max={np.max(d)} days")

print("\n--- Step 2: Running HSMM Simulation ---")

last_state = df_recent["Regime"].iloc[-1]
last_price = df_recent["SPX"].iloc[-1]
DAYS_TO_PREDICT = 60  # Let's look 2 months out

hsmm_paths = hsmm_monte_carlo(
    durations_map, 
    hsmm_trans_mat, 
    df_clean, 
    start_state=last_state, 
    start_price=last_price, 
    days_ahead=DAYS_TO_PREDICT, 
    n_sims=2000,
    random_state=42  # For reproducibility
)

# ==========================================
# 4. Visualizing Results
# ==========================================
plt.figure(figsize=(12, 6))

# Plot Mean & CI
mean_path = np.mean(hsmm_paths, axis=0)
p05 = np.percentile(hsmm_paths, 5, axis=0)
p95 = np.percentile(hsmm_paths, 95, axis=0)
p25 = np.percentile(hsmm_paths, 25, axis=0)
p75 = np.percentile(hsmm_paths, 75, axis=0)

# Fan Chart Style
plt.fill_between(range(DAYS_TO_PREDICT), p05, p95, color='blue', alpha=0.1, label="90% CI")
plt.fill_between(range(DAYS_TO_PREDICT), p25, p75, color='blue', alpha=0.2, label="50% CI")
plt.plot(mean_path, color='navy', linewidth=2, label="HSMM Mean Forecast")

# Plot a few individual "Trace" lines to show the "Sticky" behavior
# You should see distinct trends in these lines (straight lines up or down) 
# rather than random jagged noise, because the state is locked for days at a time.
plt.plot(hsmm_paths[:3].T, color='red', alpha=0.6, linewidth=1, linestyle="--", label="Single HSMM Paths")

plt.title(f"HSMM Monte Carlo: {DAYS_TO_PREDICT}-Day Forecast\n(Explicit Duration Modeling)")
plt.ylabel("S&P 500 Price")
plt.xlabel("Days Future")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.show()

# Final Numbers
expected_return = (mean_path[-1] / last_price) - 1
print(f"\nHSMM Forecast ({DAYS_TO_PREDICT} days):")
print(f"  Start Price: {last_price:.2f}")
print(f"  Exp. End Price: {mean_path[-1]:.2f} ({expected_return*100:.2f}%)")
print(f"  95% Bull Case: {p95[-1]:.2f}")
print(f"  5% Bear Case: {p05[-1]:.2f}")

# ==========================================
# 5. SAVE MODEL ARTIFACTS FOR INFERENCE
# ==========================================
import pickle
import os

# Create models directory if it doesn't exist
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n" + "=" * 60)
print("SAVING MODEL ARTIFACTS")
print("=" * 60)

# Package all artifacts needed for inference
# First, extract state returns for Monte Carlo bootstrapping
state_returns = {}
for s in range(best['K']):
    mask = df_clean['Regime'] == s
    rets = df_clean.loc[mask, 'SPX_Daily_Ret'].dropna().values
    if len(rets) > 0:
        state_returns[s] = rets
    else:
        # Fallback: use overall mean
        state_returns[s] = df_clean['SPX_Daily_Ret'].dropna().values

model_artifacts = {
    # Core HSMM model
    'hsmm_model': hsmm_best,
    
    # Preprocessing pipeline
    'scaler': scaler,
    'pca': pca,
    'n_components': n_components,
    
    # Feature configuration
    'all_features': all_features,
    'feature_groups': feature_groups,
    
    # HSMM Monte Carlo parameters
    'durations_map': durations_map,
    'transition_matrix': hsmm_trans_mat,
    'state_returns': state_returns,  # Historical returns per regime for bootstrapping
    
    # Regime diagnostics for interpretation
    'regime_diagnostics': diag,
    'regime_color_map': regime_color_map,
    'sorted_regime_ids': sorted_regime_ids,
    
    # Training metadata
    'training_end_date': df_clean['Date'].max().strftime('%Y-%m-%d'),
    'n_states': best['K'],
    'ticker_map': ticker_map,
    
    # Constants used during training
    'constants': {
        'TRADING_DAYS_YEAR': TRADING_DAYS_YEAR,
        'RSI_WINDOW': RSI_WINDOW,
        'BB_WINDOW': BB_WINDOW,
        'VOL_WINDOW': VOL_WINDOW,
        'CORR_WINDOW': CORR_WINDOW,
        'DRAWDOWN_WINDOW': DRAWDOWN_WINDOW,
        'MA_SHORT': MA_SHORT,
        'MA_LONG': MA_LONG
    }
}

# Save to pickle
model_path = os.path.join(MODEL_DIR, "hsmm_regime_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model_artifacts, f)

print(f"✓ Model saved to: {model_path}")
print(f"  - HSMM Model ({best['K']} states)")
print(f"  - Scaler (StandardScaler)")
print(f"  - PCA ({n_components} components)")
print(f"  - Feature config ({len(all_features)} features)")
print(f"  - Monte Carlo params (durations + transitions)")
print(f"  - Regime diagnostics")

# Also save the GaussianHSMM class definition for portability
class_definition_path = os.path.join(MODEL_DIR, "hsmm_class.py")
hsmm_class_code = '''
"""
GaussianHSMM Class - Required for loading the saved model.
Copy this file alongside your inference script.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

class GaussianHSMM:
    """
    Hidden Semi-Markov Model built on top of hmmlearn's GaussianHMM.
    """
    
    def __init__(self, n_components=3, covariance_type="diag",
                 duration_type="gamma", min_duration=5, max_duration=100,
                 n_iter=100, random_state=42, verbose=False):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.duration_type = duration_type
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        
        self.hmm = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            verbose=verbose
        )
        self.duration_params_ = None
    
    def fit(self, X):
        self.hmm.fit(X)
        states = self.hmm.predict(X)
        self.duration_params_ = {}
        for k in range(self.n_components):
            durations = self._extract_durations(states, k)
            if len(durations) > 0:
                mean_d = np.mean(durations)
                var_d = np.var(durations)
                scale = var_d / mean_d if mean_d > 0 else 1.0
                shape = mean_d / scale if scale > 0 else 1.0
                self.duration_params_[k] = {
                    'shape': max(shape, 0.5),
                    'scale': max(scale, 1.0),
                    'mean': mean_d,
                    'median': np.median(durations)
                }
            else:
                self.duration_params_[k] = {'shape': 2.0, 'scale': 10.0, 'mean': 20.0, 'median': 20.0}
        return self
    
    def _extract_durations(self, states, state_id):
        durations = []
        current_duration = 0
        for t in range(len(states)):
            if states[t] == state_id:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            durations.append(current_duration)
        return durations
    
    def predict(self, X):
        states_raw = self.hmm.predict(X)
        return self._enforce_min_duration(states_raw)
    
    def _enforce_min_duration(self, states):
        n = len(states)
        states_smooth = states.copy()
        i = 0
        while i < n:
            current_state = states_smooth[i]
            j = i
            while j < n and states_smooth[j] == current_state:
                j += 1
            duration = j - i
            if duration < self.min_duration and duration < n:
                prev_state = states_smooth[i-1] if i > 0 else None
                next_state = states_smooth[j] if j < n else None
                if prev_state is not None and next_state is not None:
                    replace_state = prev_state if prev_state == next_state else prev_state
                elif prev_state is not None:
                    replace_state = prev_state
                elif next_state is not None:
                    replace_state = next_state
                else:
                    replace_state = current_state
                states_smooth[i:j] = replace_state
            i = j
        return states_smooth
    
    def score(self, X):
        return self.hmm.score(X)
    
    def get_duration_stats(self):
        return pd.DataFrame(self.duration_params_).T
'''

with open(class_definition_path, 'w') as f:
    f.write(hsmm_class_code)

print(f"✓ Class definition saved to: {class_definition_path}")
print("\n" + "=" * 60)
print("MODEL SAVING COMPLETE!")
print("=" * 60)