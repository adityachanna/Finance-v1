"""
HMM Market Regime Detection API

This module provides FastAPI endpoints for Hidden Markov Model-based
market regime detection and forecasting with visualization support.
"""

import numpy as np
import pandas as pd
import joblib
import io
import base64
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================================
# Configuration & Constants
# ============================================================================

REGIME_NAMES = {
    0: "Bear",
    1: "Bull",
    2: "Neutral"
}

REGIME_COLORS = {
    0: "#ef4444",  # Red for Bear
    1: "#22c55e",  # Green for Bull
    2: "#f59e0b"   # Amber for Neutral
}

REGIME_DESCRIPTIONS = {
    0: "Bearish market conditions - characterized by falling prices, high volatility, and negative momentum",
    1: "Bullish market conditions - characterized by rising prices, low volatility, and positive momentum", 
    2: "Neutral/Sideways market - characterized by range-bound prices and moderate volatility"
}

# Confidence thresholds for regime strength classification
REGIME_CONFIDENCE_THRESHOLDS = {
    "strong": 0.70,      # >= 70% → Strong regime signal
    "moderate": 0.50,    # 50-70% → Weak/Moderate regime signal
    "uncertain": 0.0     # < 50% → Uncertain/Transition zone
}

def classify_regime_strength(max_probability: float) -> str:
    """
    Classify regime strength based on probability threshold.
    
    Returns:
        'Strong' if probability >= 70%
        'Weak' if probability 50-70%
        'Uncertain' if probability < 50%
    """
    if max_probability >= REGIME_CONFIDENCE_THRESHOLDS["strong"]:
        return "Strong"
    elif max_probability >= REGIME_CONFIDENCE_THRESHOLDS["moderate"]:
        return "Weak"
    else:
        return "Uncertain"




# ============================================================================
# Pydantic Models
# ============================================================================

class TimeHorizon(str, Enum):
    DAYS_30 = "30"
    DAYS_60 = "60"


class MarketConditionResponse(BaseModel):
    """Response model for current market condition"""
    current_regime: str = Field(..., description="Current market regime name")
    current_state: int = Field(..., description="Current regime state number")
    regime_probabilities: Dict[str, float] = Field(..., description="Probability for each regime")
    description: str = Field(..., description="Description of current market conditions")
    confidence: float = Field(..., description="Confidence level (highest probability)")
    timestamp: str = Field(..., description="Analysis timestamp")
    ticker: str = Field(..., description="Ticker symbol analyzed")
    index_value: float = Field(..., description="Current index/stock price")
    index_date: str = Field(..., description="Date of the index value")
    daily_change: float = Field(..., description="Daily price change")
    daily_change_pct: float = Field(..., description="Daily percentage change")


class ForecastDataPoint(BaseModel):
    """Single day forecast data"""
    day: int
    date: str
    bull_probability: float
    bear_probability: float
    neutral_probability: float
    most_likely_regime: str
    regime_strength: str = Field(..., description="Signal strength: Strong (≥70%), Weak (50-70%), Uncertain (<50%)")
    confidence: float = Field(..., description="Confidence level (max probability)")
    confidence_warning: Optional[str] = Field(None, description="Warning if confidence is low")


class ForecastResponse(BaseModel):
    """Response model for regime forecast"""
    current_condition: MarketConditionResponse
    horizon_days: int
    forecast: List[ForecastDataPoint]
    chart_base64: Optional[str] = Field(None, description="Base64 encoded PNG chart")
    summary: Dict[str, Any]


class HistoricalAnalysisResponse(BaseModel):
    """Response model for historical regime analysis"""
    ticker: str
    analysis_period_days: int
    historical_regimes: List[Dict[str, Any]]
    regime_distribution: Dict[str, float]
    chart_base64: Optional[str] = Field(None, description="Base64 encoded PNG chart")
    current_condition: MarketConditionResponse


class BacktestStats(BaseModel):
    """Statistics for a specific regime during backtest"""
    regime: str
    count: int
    percent_time: float
    avg_return: float
    annualized_return: float
    avg_volatility: float


class BacktestResponse(BaseModel):
    """Response model for long-term backtest"""
    ticker: str
    start_date: str
    end_date: str
    total_trading_days: int
    data: List[Dict[str, Any]]
    statistics: List[BacktestStats]


# ============================================================================
# HMM Core Functions
# ============================================================================

class HMMRegimeDetector:
    """
    Hidden Markov Model-based market regime detector.
    
    This class encapsulates all HMM-related functionality for detecting
    and forecasting market regimes.
    """
    
    def __init__(self, model_path: str = "hmm_regime_model.pkl", 
                 scaler_path: str = "feature_scaler.pkl"):
        """Initialize the detector with pre-trained model and scaler."""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.regime_names = REGIME_NAMES
        except FileNotFoundError as e:
            raise RuntimeError(f"Model files not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}")
    
    def compute_features(self, df: pd.DataFrame, window: int = 20) -> tuple:
        """
        Compute log returns and volatility features from price data.
        
        Args:
            df: DataFrame with 'Close' column
            window: Rolling window size for volatility calculation
            
        Returns:
            Tuple of (processed DataFrame, feature array)
        """
        if len(df) < window + 1:
            raise ValueError(f"Need at least {window + 1} rows of data, got {len(df)}")
        
        df = df.copy()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Return'].rolling(window).std() * np.sqrt(252)
        df = df.dropna()
        
        features = np.column_stack([
            df['Log_Return'].values,
            df['Volatility'].values
        ])
        
        return df, features
    
    def infer_current_regime(self, df: pd.DataFrame) -> tuple:
        """
        Infer the current market regime from price data.
        
        Args:
            df: DataFrame with 'Close' column
            
        Returns:
            Tuple of (current_state, state_probabilities)
        """
        df_feat, features = self.compute_features(df)
        features_scaled = self.scaler.transform(features)
        
        hidden_states = self.model.predict(features_scaled)
        state_probs = self.model.predict_proba(features_scaled)
        
        return hidden_states[-1], state_probs[-1]
    
    def get_historical_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get regime classifications for all historical data points.
        
        Args:
            df: DataFrame with 'Date' and 'Close' columns
            
        Returns:
            DataFrame with dates and their regime classifications
        """
        df_feat, features = self.compute_features(df)
        features_scaled = self.scaler.transform(features)
        
        hidden_states = self.model.predict(features_scaled)
        state_probs = self.model.predict_proba(features_scaled)
        
        result_df = df_feat.copy()
        result_df['regime_state'] = hidden_states
        result_df['regime_name'] = [self.regime_names.get(s, "Unknown") for s in hidden_states]
        
        # Add probabilities for each state
        for i in range(state_probs.shape[1]):
            result_df[f'prob_{self.regime_names.get(i, f"state_{i}")}'] = state_probs[:, i]
        
        return result_df
    
    def forecast_regime_probabilities(self, today_probs: np.ndarray, 
                                       n_days: int = 30) -> np.ndarray:
        """
        Forecast future regime probabilities using transition matrix.
        
        Args:
            today_probs: Current state probability distribution
            n_days: Number of days to forecast
            
        Returns:
            Array of shape (n_days, n_states) with future probabilities
        """
        transmat = self.model.transmat_
        probs = today_probs.copy()
        
        future_probs = np.zeros((n_days, len(probs)))
        
        for i in range(n_days):
            probs = probs @ transmat
            future_probs[i] = probs
        
        return future_probs
    
    def simulate_future_returns(self, today_state: int, 
                                 n_days: int = 30) -> tuple:
        """
        Simulate future returns using Monte Carlo sampling.
        Correctly handles inverse scaling of features.
        """
        means = self.model.means_
        covars = self.model.covars_
        transmat = self.model.transmat_
        
        state = today_state
        returns = []
        regimes = []
        
        is_diag = covars.ndim == 2
        
        for _ in range(n_days):
            # Sample from the multivariate normal distribution of the current state
            current_mean = means[state]
            current_cov = covars[state]
            
            if is_diag:
                # Diagonal covariance: independent features
                # sample ~ N(mean, std)
                sample_scaled = np.random.normal(current_mean, np.sqrt(current_cov))
            else:
                # Full covariance
                sample_scaled = np.random.multivariate_normal(current_mean, current_cov)
            
            # Inverse transform to get real values [Log_Return, Volatility]
            sample_real = self.scaler.inverse_transform([sample_scaled])[0]
            
            # Feature 0 is Log_Return
            r = sample_real[0]
            
            returns.append(r)
            regimes.append(state)
            
            state = np.random.choice(
                np.arange(len(means)),
                p=transmat[state]
            )
        
        return np.array(returns), regimes


# ============================================================================
# Data Fetching Functions
# ============================================================================

def fetch_market_data(ticker: str = "^GSPC", lookback_days: int = 120) -> pd.DataFrame:
    """
    Fetch live market data from Yahoo Finance.
    
    Args:
        ticker: Stock/index ticker symbol
        lookback_days: Number of historical days to fetch
        
    Returns:
        DataFrame with Date and Close columns
    """
    try:
        df = yf.download(
            ticker,
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        
        if df.empty:
            raise ValueError(f"No data retrieved for ticker {ticker}")
        
        df = df.reset_index()
        df_live = df[['Date', 'Close']].copy()
        
        # Handle MultiIndex columns if present
        if isinstance(df_live.columns, pd.MultiIndex):
            df_live.columns = ['Date', 'Close']
        
        df_live = df_live.sort_values('Date')
        df_live['Close'] = pd.to_numeric(df_live['Close'], errors='coerce')
        
        return df_live
    
    except Exception as e:
        raise RuntimeError(f"Error fetching data for {ticker}: {e}")


def fetch_data_range(ticker: str = "^GSPC", start_date: str = "1985-01-01") -> pd.DataFrame:
    """
    Fetch historical data from a specific start date.
    
    Args:
        ticker: Stock/index ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        
    Returns:
        DataFrame with Date and Close columns
    """
    try:
        df = yf.download(
            ticker,
            start=start_date,
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        
        if df.empty:
            raise ValueError(f"No data retrieved for ticker {ticker} from {start_date}")
        
        # Handle MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            # Try to flatten or select just the ticker level if strictly necessary
            # But yfinance usually returns simple columns if one ticker is passed and auto_adjust=True
            # Sometimes it returns (Price, Ticker) as columns.
            try:
                df.columns = df.columns.get_level_values(0)
            except:
                pass
                
        df = df.reset_index()
        
        # Normalize columns
        # yfinance lately returns 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
        # ensure we have Date and Close
        
        # If columns are just ['Open', 'High', 'Low', 'Close', 'Volume'], Date is index.
        # After reset_index, 'Date' matches.
        
        required_cols = ['Date', 'Close']
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
            
        df = df[required_cols].dropna()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        return df

    except Exception as e:
        raise RuntimeError(f"Error fetching historical data: {e}")



# ============================================================================
# Visualization Functions
# ============================================================================

def create_forecast_chart(forecast_data: np.ndarray, 
                          current_date: datetime,
                          horizon_days: int,
                          current_regime: str) -> str:
    """
    Create a forecast probability chart.
    
    Args:
        forecast_data: Array of shape (n_days, n_states) with probabilities
        current_date: Starting date for forecast
        horizon_days: Number of forecast days
        current_regime: Current market regime name
        
    Returns:
        Base64 encoded PNG image
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Set dark theme colors
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    
    dates = [current_date + timedelta(days=i+1) for i in range(horizon_days)]
    
    # Plot stacked area chart (order: Bear, Bull, Neutral for visual stacking)
    ax.stackplot(
        dates,
        forecast_data[:, 0],  # Bear (state 0)
        forecast_data[:, 1],  # Bull (state 1)
        forecast_data[:, 2],  # Neutral (state 2)
        labels=['Bear �', 'Bull �', 'Neutral ➡️'],
        colors=[REGIME_COLORS[0], REGIME_COLORS[1], REGIME_COLORS[2]],
        alpha=0.8
    )
    
    # Styling
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('Probability', fontsize=12, color='white')
    ax.set_title(
        f'Market Regime Forecast ({horizon_days} Days)\nCurrent: {current_regime}',
        fontsize=16,
        fontweight='bold',
        color='white',
        pad=20
    )
    
    ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white')
    ax.set_ylim(0, 1)
    ax.set_xlim(dates[0], dates[-1])
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def create_historical_chart(historical_df: pd.DataFrame,
                             ticker: str,
                             analysis_days: int) -> str:
    """
    Create a historical regime analysis chart with price and regime overlay.
    
    Args:
        historical_df: DataFrame with Date, Close, regime_state columns
        ticker: Ticker symbol
        analysis_days: Number of days analyzed
        
    Returns:
        Base64 encoded PNG image
    """
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    # Set dark theme
    fig.patch.set_facecolor('#1a1a2e')
    ax1.set_facecolor('#16213e')
    ax2.set_facecolor('#16213e')
    
    dates = pd.to_datetime(historical_df['Date'])
    prices = historical_df['Close'].values
    regimes = historical_df['regime_state'].values
    
    # Plot 1: Price with regime background
    ax1.plot(dates, prices, color='#00d4ff', linewidth=2, label='Price')
    
    # Add regime background colors
    for i in range(len(dates) - 1):
        ax1.axvspan(dates.iloc[i], dates.iloc[i+1], 
                    alpha=0.2, color=REGIME_COLORS.get(regimes[i], '#666'))
    
    ax1.set_ylabel('Price ($)', fontsize=12, color='white')
    ax1.set_title(
        f'{ticker} Price with Market Regime Overlay ({analysis_days} Days)',
        fontsize=16,
        fontweight='bold',
        color='white',
        pad=20
    )
    ax1.legend(loc='upper left', facecolor='#1a1a2e')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.tick_params(colors='white')
    
    # Plot 2: Regime probabilities over time
    if 'prob_Bear' in historical_df.columns:
        ax2.fill_between(dates, 0, historical_df['prob_Bear'], 
                         alpha=0.7, label='Bear', color=REGIME_COLORS[0])
        ax2.fill_between(dates, historical_df['prob_Bear'], 
                         historical_df['prob_Bear'] + historical_df['prob_Bull'],
                         alpha=0.7, label='Bull', color=REGIME_COLORS[1])
        ax2.fill_between(dates, historical_df['prob_Bear'] + historical_df['prob_Bull'], 
                         1, alpha=0.7, label='Neutral', color=REGIME_COLORS[2])
    
    ax2.set_xlabel('Date', fontsize=12, color='white')
    ax2.set_ylabel('Probability', fontsize=12, color='white')
    ax2.set_title('Regime Probabilities Over Time', fontsize=14, color='white')
    ax2.legend(loc='upper right', facecolor='#1a1a2e')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.tick_params(colors='white')
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


# ============================================================================
# Main API Functions (Callable)
# ============================================================================

def get_market_condition(ticker: str = "^GSPC") -> Dict[str, Any]:
    """
    Get the current market condition for a given ticker.
    
    This is the main callable function for programmatic use.
    
    Args:
        ticker: Stock/index ticker symbol (default: S&P 500)
        
    Returns:
        Dictionary containing current regime info and index value
    """
    detector = HMMRegimeDetector()
    df = fetch_market_data(ticker, lookback_days=120)
    
    current_state, state_probs = detector.infer_current_regime(df)
    current_regime = REGIME_NAMES.get(current_state, "Unknown")
    
    # Get latest price info
    latest_row = df.iloc[-1]
    previous_row = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    
    current_price = float(latest_row['Close'])
    previous_price = float(previous_row['Close'])
    daily_change = current_price - previous_price
    daily_change_pct = (daily_change / previous_price) * 100 if previous_price != 0 else 0
    
    # Get the date
    latest_date = latest_row['Date']
    if hasattr(latest_date, 'strftime'):
        date_str = latest_date.strftime("%Y-%m-%d")
    else:
        date_str = str(latest_date)
    
    return {
        "current_regime": current_regime,
        "current_state": int(current_state),
        "regime_probabilities": {
            REGIME_NAMES.get(i, f"State_{i}"): float(prob)
            for i, prob in enumerate(state_probs)
        },
        "description": REGIME_DESCRIPTIONS.get(current_state, "Unknown market conditions"),
        "confidence": float(max(state_probs)),
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "index_value": round(current_price, 2),
        "index_date": date_str,
        "daily_change": round(daily_change, 2),
        "daily_change_pct": round(daily_change_pct, 2)
    }


def get_regime_forecast(ticker: str = "^GSPC", 
                        horizon_days: int = 30,
                        include_chart: bool = True) -> Dict[str, Any]:
    """
    Get regime forecast for specified horizon.
    
    Args:
        ticker: Stock/index ticker symbol
        horizon_days: Forecast horizon (30 or 60 days)
        include_chart: Whether to include base64 chart
        
    Returns:
        Dictionary with forecast data and optional chart
    """
    if horizon_days not in [30, 60]:
        raise ValueError("horizon_days must be 30 or 60")
    
    detector = HMMRegimeDetector()
    df = fetch_market_data(ticker, lookback_days=180)
    
    current_state, today_probs = detector.infer_current_regime(df)
    future_probs = detector.forecast_regime_probabilities(today_probs, n_days=horizon_days)
    
    current_regime = REGIME_NAMES.get(current_state, "Unknown")
    current_date = datetime.now()
    
    # Build forecast data with confidence-aware regime classification
    forecast_data = []
    regime_transitions = 0
    strong_signal_days = 0
    weak_signal_days = 0
    uncertain_days = 0
    
    for i in range(horizon_days):
        forecast_date = current_date + timedelta(days=i+1)
        most_likely_idx = np.argmax(future_probs[i])
        max_prob = float(future_probs[i][most_likely_idx])
        
        # Classify regime strength based on confidence threshold
        regime_strength = classify_regime_strength(max_prob)
        
        # Track signal quality distribution
        if regime_strength == "Strong":
            strong_signal_days += 1
        elif regime_strength == "Weak":
            weak_signal_days += 1
        else:
            uncertain_days += 1
        
        # Count regime transitions (change in most likely regime)
        if i > 0:
            prev_regime_idx = np.argmax(future_probs[i-1])
            if most_likely_idx != prev_regime_idx:
                regime_transitions += 1
        
        # Generate confidence warning if needed
        confidence_warning = None
        if max_prob < 0.5:
            confidence_warning = "Low confidence - regime highly uncertain"
        elif max_prob < 0.7:
            confidence_warning = "Moderate confidence - signal is weakening"
        
        forecast_data.append({
            "day": i + 1,
            "date": forecast_date.strftime("%Y-%m-%d"),
            "bear_probability": float(future_probs[i][0]),    # State 0 = Bear
            "bull_probability": float(future_probs[i][1]),    # State 1 = Bull
            "neutral_probability": float(future_probs[i][2]), # State 2 = Neutral
            "most_likely_regime": REGIME_NAMES.get(most_likely_idx, "Unknown"),
            "regime_strength": regime_strength,
            "confidence": round(max_prob, 4),
            "confidence_warning": confidence_warning
        })
    
    # Calculate enhanced summary statistics
    avg_probs = np.mean(future_probs, axis=0)
    max_probs_per_day = np.max(future_probs, axis=1)
    
    # Confidence decay: compare first week avg to last week avg
    first_week_confidence = float(np.mean(max_probs_per_day[:7]))
    last_week_confidence = float(np.mean(max_probs_per_day[-7:])) if horizon_days >= 7 else first_week_confidence
    confidence_decay = first_week_confidence - last_week_confidence
    
    # Regime stability: percentage of days with strong signal and no transitions
    empirical_stability = strong_signal_days / horizon_days
    
    summary = {
        "average_bear_probability": float(avg_probs[0]),    # State 0 = Bear
        "average_bull_probability": float(avg_probs[1]),    # State 1 = Bull
        "average_neutral_probability": float(avg_probs[2]), # State 2 = Neutral
        "dominant_regime": REGIME_NAMES.get(np.argmax(avg_probs), "Unknown"),
        "regime_stability": round(empirical_stability, 4),
        "regime_transitions": regime_transitions,
        "signal_quality": {
            "strong_signal_days": strong_signal_days,
            "weak_signal_days": weak_signal_days,
            "uncertain_days": uncertain_days
        },
        "confidence_metrics": {
            "first_week_avg": round(first_week_confidence, 4),
            "last_week_avg": round(last_week_confidence, 4),
            "confidence_decay": round(confidence_decay, 4),
            "decay_warning": confidence_decay > 0.15
        }
    }
    
    result = {
        "current_condition": get_market_condition(ticker),
        "horizon_days": horizon_days,
        "forecast": forecast_data,
        "summary": summary,
        "chart_base64": None
    }
    
    if include_chart:
        result["chart_base64"] = create_forecast_chart(
            future_probs, current_date, horizon_days, current_regime
        )
    
    return result


def get_historical_analysis(ticker: str = "^GSPC",
                            analysis_days: int = 60,
                            include_chart: bool = True) -> Dict[str, Any]:
    """
    Get historical regime analysis.
    
    Args:
        ticker: Stock/index ticker symbol
        analysis_days: Number of historical days to analyze (30 or 60)
        include_chart: Whether to include base64 chart
        
    Returns:
        Dictionary with historical analysis and optional chart
    """
    if analysis_days not in [30, 60]:
        raise ValueError("analysis_days must be 30 or 60")
    
    detector = HMMRegimeDetector()
    # Fetch more data for proper calculation
    df = fetch_market_data(ticker, lookback_days=analysis_days + 60)
    
    historical_df = detector.get_historical_regimes(df)
    
    # Get last N days
    historical_df = historical_df.tail(analysis_days)
    
    # Calculate regime distribution
    regime_counts = historical_df['regime_name'].value_counts(normalize=True)
    regime_distribution = {name: float(pct) for name, pct in regime_counts.items()}
    
    # Build historical data
    historical_regimes = []
    for _, row in historical_df.iterrows():
        historical_regimes.append({
            "date": row['Date'].strftime("%Y-%m-%d") if hasattr(row['Date'], 'strftime') else str(row['Date']),
            "close_price": float(row['Close']),
            "regime": row['regime_name'],
            "regime_state": int(row['regime_state']),
            "log_return": float(row['Log_Return']),
            "volatility": float(row['Volatility'])
        })
    
    result = {
        "ticker": ticker,
        "analysis_period_days": analysis_days,
        "historical_regimes": historical_regimes,
        "regime_distribution": regime_distribution,
        "current_condition": get_market_condition(ticker),
        "chart_base64": None
    }
    
    if include_chart:
        result["chart_base64"] = create_historical_chart(
            historical_df, ticker, analysis_days
        )
    
    return result


def run_backtest_analysis(ticker: str = "^GSPC", start_date: str = "1985-01-01") -> Dict[str, Any]:
    """
    Run a long-term backtest and return data for visualization.
    """
    detector = HMMRegimeDetector()
    df = fetch_data_range(ticker, start_date)
    
    # Run HMM
    results_df = detector.get_historical_regimes(df)
    
    # Calculate Statistics
    stats_df = results_df.groupby("regime_name").agg(
        Count=("Date", "count"),
        Avg_Return=("Log_Return", "mean"),
        Avg_Volatility=("Volatility", "mean")
    )
    
    total_days = len(results_df)
    stats_list = []
    
    # Iterate through all possible regimes to ensure complete stats
    # (Even if a regime didn't appear, though unlikely in long-term)
    present_regimes = stats_df.index.tolist()
    
    for _, row in stats_df.iterrows():
        regime_name = row.name
        count = int(row["Count"])
        stats_list.append({
            "regime": regime_name,
            "count": count,
            "percent_time": round((count / total_days) * 100, 2),
            "avg_return": round(float(row["Avg_Return"]), 6),
            "annualized_return": round(float(row["Avg_Return"]) * 252 * 100, 2),
            "avg_volatility": round(float(row["Avg_Volatility"]), 4)
        })
        
    # Prepare Time-Series Data for Chart
    # We'll send a simplified list of points to keep payload manageable but sufficient for JS
    data_points = []
    for _, row in results_df.iterrows():
        data_points.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
            "regime": int(row["regime_state"]),
            "regime_name": row["regime_name"],
            "volatility": round(float(row["Volatility"]), 4)
        })
        
    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": results_df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "total_trading_days": total_days,
        "data": data_points,
        "statistics": stats_list
    }



def get_short_term_prediction(ticker: str = "^GSPC", days: int = 5) -> Dict[str, Any]:
    """
    Generate a specific 5-day short-term prediction.
    Uses Monte Carlo simulation (1000 runs) to estimate price ranges and regime stability.
    """
    detector = HMMRegimeDetector()
    df = fetch_market_data(ticker, lookback_days=180)
    
    # 1. Get Current State
    current_state, today_probs = detector.infer_current_regime(df)
    current_price = float(df.iloc[-1]['Close'])
    current_date = datetime.now()
    
    # 2. Forecast Regimes (Probabilities)
    future_probs = detector.forecast_regime_probabilities(today_probs, n_days=days)
    
    # 3. Monte Carlo Simulation for Price Targets
    # Run 1000 simulations to get a distribution of possible future prices
    simulations = 1000
    final_prices = []
    
    for _ in range(simulations):
        sim_returns, _ = detector.simulate_future_returns(current_state, n_days=days)
        # Calculate cumulative return: (1+r1)*(1+r2)...
        # Note: log returns are additive, so exp(sum(log_returns)) gives total return multiplier
        total_return = np.exp(np.sum(sim_returns))
        final_prices.append(current_price * total_return)
        
    final_prices = np.sort(final_prices)
    
    # Calculate Percentiles
    p5 = float(np.percentile(final_prices, 5))   # Worst case (95% confidence)
    p50 = float(np.median(final_prices))         # Expected
    p95 = float(np.percentile(final_prices, 95)) # Best case
    
    # 4. Construct Day-by-Day Forecast with confidence-aware classification
    daily_forecast = []
    for i in range(days):
        day_date = current_date + timedelta(days=i+1)
        most_likely_regime_idx = np.argmax(future_probs[i])
        max_prob = float(future_probs[i][most_likely_regime_idx])
        regime_strength = classify_regime_strength(max_prob)
        
        daily_forecast.append({
            "day": i + 1,
            "date": day_date.strftime("%Y-%m-%d"),
            "regime": REGIME_NAMES.get(most_likely_regime_idx, "Unknown"),
            "regime_strength": regime_strength,
            "confidence": round(max_prob, 2),
            "bull_prob": round(float(future_probs[i][1]), 2),    # Index 1 is Bull
            "bear_prob": round(float(future_probs[i][0]), 2),    # Index 0 is Bear
            "neutral_prob": round(float(future_probs[i][2]), 2)  # Index 2 is Neutral
        })
    
    # Determine overall trend strength
    avg_confidence = np.mean([f["confidence"] for f in daily_forecast])
    trend_direction = "Bullish" if p50 > current_price else "Bearish/Neutral"
    trend_strength = classify_regime_strength(avg_confidence)
    
    return {
        "ticker": ticker,
        "prediction_days": days,
        "current_price": round(current_price, 2),
        "expected_price_5d": round(p50, 2),
        "price_range_5d": {
            "low_estimate": round(p5, 2),
            "high_estimate": round(p95, 2)
        },
        "daily_forecast": daily_forecast,
        "summary": f"{trend_strength} {trend_direction} trend expected",
        "average_confidence": round(avg_confidence, 2)
    }


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="HMM Market Regime API",
    description="""
    ## Hidden Markov Model Market Regime Detection API
    
    This API provides endpoints for detecting and forecasting market regimes
    using a pre-trained Hidden Markov Model.
    
    ### Features:
    - **Current Market Condition**: Get real-time market regime classification
    - **Regime Forecast**: Forecast regime probabilities for 30/60 days
    - **Historical Analysis**: Analyze past 30/60 days with regime overlays
    - **Visualization**: Charts returned as base64-encoded PNG images
    
    ### Market Regimes:
    - **Bull**: Rising prices, low volatility, positive momentum
    - **Bear**: Falling prices, high volatility, negative momentum
    - **Neutral**: Range-bound prices, moderate volatility
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "HMM Market Regime API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/market-condition", 
         response_model=MarketConditionResponse,
         tags=["Market Analysis"])
async def api_market_condition(
    ticker: str = Query(default="^GSPC", description="Ticker symbol (e.g., ^GSPC for S&P 500)")
):
    """
    Get the current market condition and regime classification.
    
    Returns the current regime (Bull/Bear/Neutral), probabilities for each regime,
    and a description of current market conditions.
    """
    try:
        result = get_market_condition(ticker)
        return MarketConditionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/{horizon}", 
         response_model=ForecastResponse,
         tags=["Forecasting"])
async def api_regime_forecast(
    horizon: TimeHorizon,
    ticker: str = Query(default="^GSPC", description="Ticker symbol"),
    include_chart: bool = Query(default=True, description="Include base64 chart")
):
    """
    Get regime probability forecast for 30 or 60 days.
    
    Returns daily regime probabilities and an optional stacked area chart
    showing the evolution of regime probabilities over the forecast horizon.
    """
    try:
        horizon_days = int(horizon.value)
        result = get_regime_forecast(ticker, horizon_days, include_chart)
        return ForecastResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/historical/{days}", 
         response_model=HistoricalAnalysisResponse,
         tags=["Historical Analysis"])
async def api_historical_analysis(
    days: TimeHorizon,
    ticker: str = Query(default="^GSPC", description="Ticker symbol"),
    include_chart: bool = Query(default=True, description="Include base64 chart")
):
    """
    Get historical regime analysis for the last 30 or 60 days.
    
    Returns historical regime classifications, price data, and an optional
    chart showing price with regime overlay and probability evolution.
    """
    try:
        analysis_days = int(days.value)
        result = get_historical_analysis(ticker, analysis_days, include_chart)
        return HistoricalAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest", 
         response_model=BacktestResponse,
         tags=["Historical Analysis"])
async def api_backtest(
    ticker: str = Query(default="^GSPC", description="Ticker symbol"),
    start_date: str = Query(default="1985-01-01", description="Start date (YYYY-MM-DD)")
):
    """
    Get long-term backtest data for custom charting.
    
    Returns daily price, regime, and volatility data from the start_date to present,
    along with aggregate regime statistics.
    """
    try:
        result = run_backtest_analysis(ticker, start_date)
        return BacktestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/combined-analysis", tags=["Market Analysis"])
async def api_combined_analysis(
    ticker: str = Query(default="^GSPC", description="Ticker symbol"),
    include_charts: bool = Query(default=True, description="Include base64 charts")
):
    """
    Get comprehensive market analysis including current condition,
    30-day forecast, 60-day forecast, and historical analysis.
    
    This endpoint combines all analysis types into a single response.
    Includes index value and price information.
    """
    try:
        # Fetch market data once for index info
        df = fetch_market_data(ticker, lookback_days=120)
        
        # Get index statistics
        latest_price = float(df.iloc[-1]['Close'])
        previous_price = float(df.iloc[-2]['Close']) if len(df) > 1 else latest_price
        daily_change = latest_price - previous_price
        daily_change_pct = (daily_change / previous_price) * 100 if previous_price != 0 else 0
        
        # Calculate period statistics
        prices = df['Close'].values
        high_30 = float(prices[-30:].max()) if len(prices) >= 30 else float(prices.max())
        low_30 = float(prices[-30:].min()) if len(prices) >= 30 else float(prices.min())
        high_60 = float(prices[-60:].max()) if len(prices) >= 60 else float(prices.max())
        low_60 = float(prices[-60:].min()) if len(prices) >= 60 else float(prices.min())
        
        # Get the date
        latest_date = df.iloc[-1]['Date']
        if hasattr(latest_date, 'strftime'):
            date_str = latest_date.strftime("%Y-%m-%d")
        else:
            date_str = str(latest_date)
        
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "index_info": {
                "current_value": round(latest_price, 2),
                "previous_close": round(previous_price, 2),
                "daily_change": round(daily_change, 2),
                "daily_change_pct": round(daily_change_pct, 2),
                "price_date": date_str,
                "high_30_days": round(high_30, 2),
                "low_30_days": round(low_30, 2),
                "high_60_days": round(high_60, 2),
                "low_60_days": round(low_60, 2),
                "distance_from_30d_high_pct": round(((high_30 - latest_price) / high_30) * 100, 2),
                "distance_from_30d_low_pct": round(((latest_price - low_30) / low_30) * 100, 2)
            },
            "current_condition": get_market_condition(ticker),
            "forecast_30_days": get_regime_forecast(ticker, 30, include_charts),
            "forecast_60_days": get_regime_forecast(ticker, 60, include_charts),
            "historical_30_days": get_historical_analysis(ticker, 30, include_charts),
            "historical_60_days": get_historical_analysis(ticker, 60, include_charts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting HMM Market Regime API...")
    print("📊 Endpoints available:")
    print("   - GET /api/market-condition - Current market regime")
    print("   - GET /api/forecast/{30|60} - Regime forecast")
    print("   - GET /api/historical/{30|60} - Historical analysis")
    print("   - GET /api/combined-analysis - Full analysis")
    print("📖 Documentation: http://localhost:8000/docs")
    
    uvicorn.run("hmm_api:app", host="0.0.0.0", port=8000, reload=True)
