"""
HSMM Market Regime Inference Script
====================================
This script loads the pre-trained HSMM model and performs:
1. Current regime detection
2. Monte Carlo simulation for price forecasting

Usage:
    python hsmm_inference.py

Requirements:
    - Pre-trained model at models/hsmm_regime_model.pkl
    - Run nice.py first to train and save the model
"""

import pickle
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import GaussianHSMM class - required for pickle to deserialize the model
from models.hsmm_class import GaussianHSMM

# ============================================
# CUSTOM UNPICKLER TO HANDLE CLASS RESOLUTION
# ============================================

class HSMMUnpickler(pickle.Unpickler):
    """
    Custom unpickler that redirects __main__.GaussianHSMM 
    to models.hsmm_class.GaussianHSMM.
    """
    def find_class(self, module, name):
        if name == 'GaussianHSMM':
            return GaussianHSMM
        return super().find_class(module, name)

# ============================================
# 1. LOAD MODEL AND ARTIFACTS
# ============================================

def load_model(model_path: str = "models/hsmm_regime_model.pkl") -> dict:
    """Load the pre-trained HSMM model and all artifacts."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run nice.py first to train and save the model."
        )
    
    # Use custom unpickler to handle GaussianHSMM class resolution
    with open(model_path, 'rb') as f:
        artifacts = HSMMUnpickler(f).load()
    
    print("=" * 60)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"  States: {artifacts['n_states']}")
    print(f"  Features: {len(artifacts['all_features'])}")
    print(f"  PCA Components: {artifacts['n_components']}")
    print(f"  Training End: {artifacts['training_end_date']}")
    
    return artifacts


# ============================================
# 2. FEATURE ENGINEERING (Same as Training)
# ============================================

def calc_rsi(series, window=14):
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_bollinger_width(series, window=20):
    """Calculate Bollinger Band width as (upper - lower) / ma."""
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + (2 * std)
    lower = ma - (2 * std)
    return (upper - lower) / ma

def realized_vol(series, window=21, trading_days=252):
    """Calculate annualized realized volatility."""
    return series.rolling(window).std() * np.sqrt(trading_days)

def calc_drawdown(series, window=126):
    """Calculate rolling drawdown from peak."""
    roll_max = series.rolling(window, min_periods=1).max()
    return series / roll_max - 1.0


def fetch_and_engineer_features(
    ticker_map: dict,
    start_date: str = None,
    constants: dict = None
) -> pd.DataFrame:
    """
    Fetch fresh market data and engineer all features.
    
    Parameters:
    -----------
    ticker_map : dict
        Mapping of Yahoo tickers to friendly names
    start_date : str
        Start date for data fetch (needs buffer for rolling calculations)
    constants : dict
        Constants from training (windows, etc.)
    
    Returns:
    --------
    pd.DataFrame with all engineered features
    """
    # Default constants if not provided
    if constants is None:
        constants = {
            'TRADING_DAYS_YEAR': 252,
            'RSI_WINDOW': 14,
            'BB_WINDOW': 20,
            'VOL_WINDOW': 21,
            'CORR_WINDOW': 63,
            'DRAWDOWN_WINDOW': 126,
            'MA_SHORT': 50,
            'MA_LONG': 200
        }
    
    # Calculate start date with buffer for 200-day MA
    if start_date is None:
        from datetime import datetime, timedelta
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    
    print(f"\nFetching data from {start_date}...")
    
    # Fetch data
    tickers = list(ticker_map.keys())
    raw_data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)
    
    # Handle MultiIndex columns
    if isinstance(raw_data.columns, pd.MultiIndex):
        try:
            df = raw_data['Adj Close'].copy()
        except KeyError:
            df = raw_data['Close'].copy()
    else:
        df = raw_data.copy()
    
    df = df.rename(columns=ticker_map).ffill().dropna().reset_index()
    if 'Date' not in df.columns:
        df = df.rename(columns={'index': 'Date'})
    
    print(f"Data range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # === Feature Engineering ===
    for col in ["SPX", "NDX", "RUT"]:
        # Multi-timeframe returns
        df[f"{col}_Ret_5D"] = df[col].pct_change(5)
        df[f"{col}_Ret_21D"] = df[col].pct_change(21)
        df[f"{col}_Ret_63D"] = df[col].pct_change(63)
        df[f"{col}_Ret_126D"] = df[col].pct_change(126)
        
        # Technical indicators
        df[f"{col}_RSI"] = calc_rsi(df[col], constants['RSI_WINDOW'])
        df[f"{col}_BB_Width"] = calc_bollinger_width(df[col], constants['BB_WINDOW'])
        
        # Trend metrics
        df[f"{col}_Dist_MA50"] = (df[col] / df[col].rolling(constants['MA_SHORT']).mean()) - 1.0
        df[f"{col}_Dist_MA200"] = (df[col] / df[col].rolling(constants['MA_LONG']).mean()) - 1.0
        
        # Daily returns
        df[f"{col}_Daily_Ret"] = df[col].pct_change()
    
    # Volatility metrics
    df["SPX_RealVol"] = realized_vol(df["SPX_Daily_Ret"], constants['VOL_WINDOW'], constants['TRADING_DAYS_YEAR'])
    df["NDX_RealVol"] = realized_vol(df["NDX_Daily_Ret"], constants['VOL_WINDOW'], constants['TRADING_DAYS_YEAR'])
    df["SPX_Skew_63D"] = df["SPX_Daily_Ret"].rolling(63).skew()
    
    # VIX dynamics
    df["VIX_vs_MA50"] = df["VIX"] / df["VIX"].rolling(constants['MA_SHORT']).mean()
    df["VIX_Change_5D"] = df["VIX"].diff(5)
    df["Vol_Risk_Premium"] = df["VIX"] - (df["SPX_RealVol"] * 100)
    
    # Cross-asset relationships
    window_corr = constants['CORR_WINDOW']
    df["Corr_NDX_SPX"] = df["NDX_Daily_Ret"].rolling(window_corr).corr(df["SPX_Daily_Ret"])
    df["Corr_RUT_SPX"] = df["RUT_Daily_Ret"].rolling(window_corr).corr(df["SPX_Daily_Ret"])
    
    cov_rut = df["RUT_Daily_Ret"].rolling(window_corr).cov(df["SPX_Daily_Ret"])
    var_spx = df["SPX_Daily_Ret"].rolling(window_corr).var()
    df["Beta_RUT_SPX"] = cov_rut / var_spx
    
    df["Ratio_NDX_SPX"] = df["NDX"] / df["SPX"]
    df["Ratio_NDX_SPX_Trend"] = df["Ratio_NDX_SPX"].pct_change(63)
    
    # Drawdowns
    df["SPX_Drawdown"] = calc_drawdown(df["SPX"], constants['DRAWDOWN_WINDOW'])
    df["NDX_Drawdown"] = calc_drawdown(df["NDX"], constants['DRAWDOWN_WINDOW'])
    
    # Clean
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    
    print(f"Processed samples: {len(df_clean)}")
    
    return df_clean


# ============================================
# 3. REGIME INFERENCE
# ============================================

def infer_current_regime(
    artifacts: dict,
    df: pd.DataFrame = None,
    return_history: bool = False
) -> dict:
    """
    Infer the current market regime using the pre-trained model.
    
    Parameters:
    -----------
    artifacts : dict
        Loaded model artifacts from load_model()
    df : pd.DataFrame, optional
        Pre-processed dataframe. If None, will fetch fresh data.
    return_history : bool
        If True, return regime history for all data points
    
    Returns:
    --------
    dict with current regime info and optionally full history
    """
    # Fetch data if not provided
    if df is None:
        df = fetch_and_engineer_features(
            artifacts['ticker_map'],
            constants=artifacts.get('constants')
        )
    
    # Extract features
    all_features = artifacts['all_features']
    X = df[all_features].values
    
    # Apply preprocessing pipeline
    X_scaled = artifacts['scaler'].transform(X)
    X_pca = artifacts['pca'].transform(X_scaled)[:, :artifacts['n_components']]
    
    # Predict regimes
    regimes = artifacts['hsmm_model'].predict(X_pca)
    df['Regime'] = regimes
    
    # Get current state
    current_regime = regimes[-1]
    current_date = df['Date'].iloc[-1]
    current_price = df['SPX'].iloc[-1]
    current_vix = df['VIX'].iloc[-1]
    
    # Get regime label from diagnostics
    diag = artifacts['regime_diagnostics']
    regime_info = diag[diag['State'] == current_regime].iloc[0]
    
    result = {
        'date': current_date.strftime('%Y-%m-%d'),
        'regime_id': int(current_regime),
        'regime_label': regime_info['Label'],
        'spx_price': float(current_price),
        'vix': float(current_vix),
        'regime_stats': {
            'annual_return_pct': float(regime_info['Ann_Ret_%']),
            'annual_vol_pct': float(regime_info['Ann_Vol_%']),
            'avg_duration_days': float(regime_info['Avg_Duration']),
            'avg_drawdown_pct': float(regime_info['Avg_DD_%'])
        }
    }
    
    if return_history:
        result['history'] = df[['Date', 'SPX', 'VIX', 'Regime']].copy()
    
    return result


# ============================================
# 4. MONTE CARLO SIMULATION
# ============================================

def hsmm_monte_carlo_forecast(
    artifacts: dict,
    start_state: int,
    start_price: float,
    days_ahead: int = 60,
    n_sims: int = 2000,
    random_state: int = 42
) -> dict:
    """
    Run Monte Carlo simulation using HSMM duration properties.
    
    Parameters:
    -----------
    artifacts : dict
        Loaded model artifacts
    start_state : int
        Current regime state to start simulation from
    start_price : float
        Current S&P 500 price
    days_ahead : int
        Number of days to forecast
    n_sims : int
        Number of simulation paths
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict with forecast statistics and paths
    """
    np.random.seed(random_state)
    
    durations_map = artifacts['durations_map']
    trans_probs = artifacts['transition_matrix']
    
    # Get historical returns by regime for bootstrapping
    # Better check: verify state_returns exists and has non-empty arrays
    state_returns_valid = False
    if 'state_returns' in artifacts:
        sr = artifacts['state_returns']
        if sr and isinstance(sr, dict) and len(sr) > 0:
            # Check if at least one state has returns
            total_returns = sum(len(v) for v in sr.values() if hasattr(v, '__len__'))
            if total_returns > 0:
                state_returns_valid = True
                state_returns = sr
                print(f"  Using actual historical returns from training data")
                print(f"    States with returns: {list(sr.keys())}")
                print(f"    Total return samples: {total_returns}")
    
    if not state_returns_valid:
        print("  Warning: No historical returns in artifacts. Using synthetic returns.")
        print(f"    'state_returns' in artifacts: {'state_returns' in artifacts}")
        if 'state_returns' in artifacts:
            sr = artifacts['state_returns']
            print(f"    Type: {type(sr)}, Value: {sr if not sr else 'non-empty'}")
        
        # Create synthetic returns based on regime stats
        diag = artifacts['regime_diagnostics']
        state_returns = {}
        for _, row in diag.iterrows():
            state = int(row['State'])
            mean_daily = row['Ann_Ret_%'] / 100 / 252
            std_daily = row['Ann_Vol_%'] / 100 / np.sqrt(252)
            # Generate synthetic returns
            state_returns[state] = np.random.normal(mean_daily, std_daily, 1000)
    
    future_paths = np.zeros((n_sims, days_ahead))
    
    for sim_i in range(n_sims):
        current_price = start_price
        current_state = start_state
        day_count = 0
        path = []
        
        while day_count < days_ahead:
            # Sample duration
            possible_durations = durations_map.get(current_state, [5])
            if len(possible_durations) > 0:
                stay_duration = np.random.choice(possible_durations)
            else:
                stay_duration = 5
            
            # Simulate for that duration
            for t in range(stay_duration):
                if day_count >= days_ahead:
                    break
                
                # Sample return
                returns = state_returns.get(current_state, np.array([0.0]))
                r = np.random.choice(returns)
                current_price = current_price * (1 + r)
                path.append(current_price)
                day_count += 1
            
            # Force state switch
            probs = trans_probs[current_state]
            if probs.sum() > 0:
                next_state = np.random.choice(len(probs), p=probs)
            else:
                # Fallback: stay in current state
                next_state = current_state
            current_state = next_state
        
        future_paths[sim_i, :] = path
    
    # Calculate statistics
    mean_path = np.mean(future_paths, axis=0)
    p05 = np.percentile(future_paths, 5, axis=0)
    p25 = np.percentile(future_paths, 25, axis=0)
    p50 = np.percentile(future_paths, 50, axis=0)
    p75 = np.percentile(future_paths, 75, axis=0)
    p95 = np.percentile(future_paths, 95, axis=0)
    
    expected_return = (mean_path[-1] / start_price - 1) * 100
    
    return {
        'start_price': start_price,
        'days_ahead': days_ahead,
        'n_sims': n_sims,
        'paths': future_paths,
        'mean_path': mean_path,
        'percentiles': {
            'p05': p05,
            'p25': p25,
            'p50': p50,
            'p75': p75,
            'p95': p95
        },
        'final_stats': {
            'expected_price': float(mean_path[-1]),
            'expected_return_pct': float(expected_return),
            'bull_case_p95': float(p95[-1]),
            'bear_case_p05': float(p05[-1]),
            'median_price': float(p50[-1])
        }
    }


def plot_forecast(forecast: dict, show: bool = True) -> None:
    """Plot the Monte Carlo forecast fan chart."""
    plt.figure(figsize=(12, 6))
    
    days = range(forecast['days_ahead'])
    p = forecast['percentiles']
    
    # Fan chart
    plt.fill_between(days, p['p05'], p['p95'], color='blue', alpha=0.1, label="90% CI")
    plt.fill_between(days, p['p25'], p['p75'], color='blue', alpha=0.2, label="50% CI")
    plt.plot(forecast['mean_path'], color='navy', linewidth=2, label="Expected Path")
    plt.plot(p['p50'], color='green', linewidth=1, linestyle='--', label="Median Path")
    
    # Sample paths
    plt.plot(forecast['paths'][:3].T, color='red', alpha=0.4, linewidth=0.5)
    
    plt.title(f"HSMM Monte Carlo: {forecast['days_ahead']}-Day Forecast\n"
              f"Expected: ${forecast['final_stats']['expected_price']:.0f} "
              f"({forecast['final_stats']['expected_return_pct']:+.1f}%)")
    plt.ylabel("S&P 500 Price")
    plt.xlabel("Days Forward")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if show:
        plt.show()


# ============================================
# 5. MAIN INFERENCE FUNCTION
# ============================================

def run_inference(
    days_ahead: int = 60,
    n_sims: int = 2000,
    plot: bool = True
) -> dict:
    """
    Complete inference pipeline: load model, detect regime, forecast.
    
    Returns:
    --------
    dict with all inference results
    """
    # Load model
    artifacts = load_model()
    
    # Infer current regime
    print("\n" + "=" * 60)
    print("REGIME DETECTION")
    print("=" * 60)
    
    regime_result = infer_current_regime(artifacts, return_history=True)
    
    print(f"\nCurrent Market State (as of {regime_result['date']}):")
    print(f"  Regime: {regime_result['regime_id']} - {regime_result['regime_label']}")
    print(f"  S&P 500: ${regime_result['spx_price']:.2f}")
    print(f"  VIX: {regime_result['vix']:.2f}")
    print(f"\nRegime Characteristics:")
    stats = regime_result['regime_stats']
    print(f"  Annual Return: {stats['annual_return_pct']:.1f}%")
    print(f"  Annual Volatility: {stats['annual_vol_pct']:.1f}%")
    print(f"  Avg Duration: {stats['avg_duration_days']:.0f} days")
    print(f"  Avg Drawdown: {stats['avg_drawdown_pct']:.1f}%")
    
    # Run Monte Carlo forecast
    print("\n" + "=" * 60)
    print("MONTE CARLO FORECAST")
    print("=" * 60)
    
    forecast = hsmm_monte_carlo_forecast(
        artifacts,
        start_state=regime_result['regime_id'],
        start_price=regime_result['spx_price'],
        days_ahead=days_ahead,
        n_sims=n_sims
    )
    
    print(f"\n{days_ahead}-Day Forecast ({n_sims} simulations):")
    print(f"  Start Price: ${forecast['start_price']:.2f}")
    print(f"  Expected Price: ${forecast['final_stats']['expected_price']:.2f}")
    print(f"  Expected Return: {forecast['final_stats']['expected_return_pct']:+.2f}%")
    print(f"  Bull Case (95%): ${forecast['final_stats']['bull_case_p95']:.2f}")
    print(f"  Bear Case (5%): ${forecast['final_stats']['bear_case_p05']:.2f}")
    
    if plot:
        plot_forecast(forecast)
    
    return {
        'regime': regime_result,
        'forecast': forecast,
        'artifacts': artifacts
    }


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HSMM MARKET REGIME INFERENCE")
    print("=" * 60)
    
    try:
        results = run_inference(days_ahead=60, n_sims=2000, plot=True)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run 'python nice.py' first to train and save the model.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during inference: {e}")
        raise
