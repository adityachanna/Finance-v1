"""
HSMM Market Regime API Module
=============================
Provides API-friendly functions for market regime detection and forecasting
using the Hidden Semi-Markov Model trained in nice.py.

This module wraps hsmm_inference.py functions for use in FastAPI endpoints.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

# CRITICAL: Import the GaussianHSMM class BEFORE any pickle operations
# This makes the class available in the namespace for deserialization
from models.hsmm_class import GaussianHSMM

# Import core inference functions (but NOT load_model - we do that ourselves)
from hsmm_inference import (
    fetch_and_engineer_features,
    infer_current_regime,
    hsmm_monte_carlo_forecast
)

# ============================================
# HELPER: CONVERT NUMPY TYPES FOR JSON
# ============================================

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    else:
        return obj
# CUSTOM UNPICKLER TO HANDLE CLASS RESOLUTION
# ============================================

class HSMMUnpickler(pickle.Unpickler):
    """
    Custom unpickler that redirects __main__.GaussianHSMM 
    to models.hsmm_class.GaussianHSMM.
    
    This is necessary because when nice.py pickles the model, the class
    is referenced as __main__.GaussianHSMM. When loading from a different
    script, we need to redirect that reference.
    """
    def find_class(self, module, name):
        if name == 'GaussianHSMM':
            # Always use the class from models.hsmm_class
            return GaussianHSMM
        return super().find_class(module, name)


# ============================================
# SINGLETON MODEL LOADER (Cache the model)
# ============================================

_cached_artifacts = None
MODEL_PATH = "models/hsmm_regime_model.pkl"

def get_model_artifacts() -> dict:
    """
    Load and cache model artifacts.
    Uses singleton pattern to avoid reloading on every request.
    
    Uses custom unpickler to handle GaussianHSMM class resolution
    regardless of the original pickle context.
    """
    global _cached_artifacts
    if _cached_artifacts is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please run nice.py first to train and save the model."
            )
        
        # Use custom unpickler to handle class resolution
        with open(MODEL_PATH, 'rb') as f:
            _cached_artifacts = HSMMUnpickler(f).load()
        
        print(f"[HSMM API] Model loaded: {_cached_artifacts['n_states']} states, "
              f"trained until {_cached_artifacts['training_end_date']}")
    
    return _cached_artifacts


# ============================================
# API FUNCTIONS
# ============================================

def get_market_condition(ticker: str = "SPX") -> Dict[str, Any]:
    """
    Get the current HSMM-based market regime.
    
    Optimized for dashboard display with:
    - Clean numbers (2 decimal places)
    - Risk level indicator (Low/Medium/High/Extreme)
    - Trend direction
    - Human-readable summary
    """
    try:
        artifacts = get_model_artifacts()
        
        # Infer current regime WITH history to calculate days in regime
        regime_result = infer_current_regime(artifacts, return_history=True)
        
        # Get regime diagnostics
        diag = artifacts['regime_diagnostics']
        current_regime_info = diag[diag['State'] == regime_result['regime_id']].iloc[0]
        
        # Calculate days in current regime
        history = regime_result['history']
        current_regime_id = regime_result['regime_id']
        days_in_regime = 0
        for i in range(len(history) - 1, -1, -1):
            if history.iloc[i]['Regime'] == current_regime_id:
                days_in_regime += 1
            else:
                break
        
        # Determine risk level based on regime characteristics
        ann_ret = current_regime_info['Ann_Ret_%']
        ann_vol = current_regime_info['Ann_Vol_%']
        avg_vix = current_regime_info['Avg_VIX']
        
        if ann_ret < -20 or avg_vix > 30:
            risk_level = "Extreme"
            risk_color = "red"
        elif ann_ret < -5 or avg_vix > 22:
            risk_level = "High"
            risk_color = "orange"
        elif ann_vol > 15 or avg_vix > 18:
            risk_level = "Medium"
            risk_color = "yellow"
        else:
            risk_level = "Low"
            risk_color = "green"
        
        # Determine trend based on recent price movement
        if len(history) >= 5:
            recent_prices = history['SPX'].tail(5).values
            price_change = (recent_prices[-1] / recent_prices[0] - 1) * 100
            if price_change > 1:
                trend = "Improving"
                trend_icon = "↗"
            elif price_change < -1:
                trend = "Declining"
                trend_icon = "↘"
            else:
                trend = "Stable"
                trend_icon = "→"
        else:
            trend = "Unknown"
            trend_icon = "?"
            price_change = 0
        
        # Generate summary text
        label = regime_result['regime_label']
        summary = f"Market is in {label} regime ({days_in_regime} days). "
        if risk_level in ["Low", "Medium"]:
            summary += f"Conditions are favorable with {round(ann_ret, 1)}% expected annual return."
        elif risk_level == "High":
            summary += f"Elevated volatility detected. Exercise caution."
        else:
            summary += f"High-stress environment. Risk management is critical."
        
        # Build clean response
        response = {
            "ticker": "SPX",
            "date": regime_result['date'],
            "regime": {
                "id": regime_result['regime_id'],
                "label": regime_result['regime_label'],
                "days_in_regime": days_in_regime,
                "avg_duration": round(current_regime_info['Avg_Duration'], 0)
            },
            "market_data": {
                "spx_price": round(regime_result['spx_price'], 2),
                "vix": round(regime_result['vix'], 2),
                "5d_change_pct": round(price_change, 2)
            },
            "risk_assessment": {
                "level": risk_level,
                "color": risk_color,
                "trend": trend,
                "trend_icon": trend_icon
            },
            "regime_stats": {
                "annual_return_pct": round(current_regime_info['Ann_Ret_%'], 2),
                "annual_volatility_pct": round(current_regime_info['Ann_Vol_%'], 2),
                "avg_drawdown_pct": round(current_regime_info['Avg_DD_%'], 2),
                "avg_vix": round(current_regime_info['Avg_VIX'], 2)
            },
            "summary": summary,
            "model_info": {
                "type": "HSMM",
                "states": int(artifacts['n_states']),
                "training_end": artifacts['training_end_date']
            }
        }
        
        return convert_numpy_types(response)
    except FileNotFoundError as e:
        raise Exception(f"Model not found. Please run nice.py first to train the model. Error: {e}")
    except Exception as e:
        raise Exception(f"Error getting market condition: {e}")


def get_regime_forecast(
    ticker: str = "SPX",
    horizon_days: int = 60,
    n_simulations: int = 2000,
    include_paths: bool = False
) -> Dict[str, Any]:
    """
    Get HSMM-based Monte Carlo forecast for market regime.
    
    Parameters:
    -----------
    ticker : str
        Ignored (model uses SPX), kept for API compatibility
    horizon_days : int
        Forecast horizon (default: 60 days)
    n_simulations : int
        Number of Monte Carlo paths (default: 2000)
    include_paths : bool
        If True, include sample paths in response (warning: large)
    
    Returns:
    --------
    dict with forecast statistics and price projections
    """
    try:
        artifacts = get_model_artifacts()
        
        # Get current regime first
        regime_result = infer_current_regime(artifacts, return_history=False)
        
        # Run Monte Carlo forecast
        forecast = hsmm_monte_carlo_forecast(
            artifacts,
            start_state=regime_result['regime_id'],
            start_price=regime_result['spx_price'],
            days_ahead=horizon_days,
            n_sims=n_simulations,
            random_state=42
        )
        
        # Build response
        response = {
            "ticker": "SPX",
            "forecast_date": regime_result['date'],
            "current_regime": {
                "id": regime_result['regime_id'],
                "label": regime_result['regime_label']
            },
            "horizon_days": horizon_days,
            "n_simulations": n_simulations,
            "starting_price": forecast['start_price'],
            "forecast": {
                "expected_price": round(forecast['final_stats']['expected_price'], 2),
                "expected_return_pct": round(forecast['final_stats']['expected_return_pct'], 2),
                "median_price": round(forecast['final_stats']['median_price'], 2),
                "bull_case_95pct": round(forecast['final_stats']['bull_case_p95'], 2),
                "bear_case_5pct": round(forecast['final_stats']['bear_case_p05'], 2)
            },
            "confidence_intervals": {
                "90_pct": {
                    "lower": round(float(forecast['percentiles']['p05'][-1]), 2),
                    "upper": round(float(forecast['percentiles']['p95'][-1]), 2)
                },
                "50_pct": {
                    "lower": round(float(forecast['percentiles']['p25'][-1]), 2),
                    "upper": round(float(forecast['percentiles']['p75'][-1]), 2)
                }
            },
            "model_info": {
                "type": "HSMM Monte Carlo with Duration Modeling",
                "methodology": "Bootstrap historical returns per regime with Semi-Markov transitions"
            }
        }
        
        # Optionally include mean path for charting
        if include_paths:
            # Convert to list for JSON serialization (sample every 5 days for efficiency)
            response["mean_path"] = {
                "days": list(range(0, horizon_days, 5)),
                "prices": [round(float(p), 2) for p in forecast['mean_path'][::5]]
            }
            # Include 3 sample paths
            response["sample_paths"] = [
                [round(float(p), 2) for p in path[::5]] 
                for path in forecast['paths'][:3]
            ]
        
        return convert_numpy_types(response)
        
    except FileNotFoundError as e:
        raise Exception(f"Model not found. Please run nice.py first. Error: {e}")
    except Exception as e:
        raise Exception(f"Error generating forecast: {e}")


def get_short_term_prediction(
    ticker: str = "SPX",
    days: int = 5
) -> Dict[str, Any]:
    """
    Get a short-term prediction (5-10 days) with daily price estimates.
    
    This is useful for near-term trading decisions.
    
    Returns:
    --------
    dict with daily expected prices and regime path
    """
    try:
        artifacts = get_model_artifacts()
        
        # Get current regime
        regime_result = infer_current_regime(artifacts, return_history=False)
        
        # Run Monte Carlo for short term
        forecast = hsmm_monte_carlo_forecast(
            artifacts,
            start_state=regime_result['regime_id'],
            start_price=regime_result['spx_price'],
            days_ahead=days,
            n_sims=1000,  # Fewer sims okay for short term
            random_state=42
        )
        
        # Build daily predictions
        daily_predictions = []
        for i in range(days):
            daily_predictions.append({
                "day": i + 1,
                "expected_price": round(float(forecast['mean_path'][i]), 2),
                "range_low": round(float(forecast['percentiles']['p25'][i]), 2),
                "range_high": round(float(forecast['percentiles']['p75'][i]), 2)
            })
        
        response = {
            "ticker": "SPX",
            "prediction_date": regime_result['date'],
            "current_state": {
                "regime_id": regime_result['regime_id'],
                "regime_label": regime_result['regime_label'],
                "current_price": regime_result['spx_price'],
                "vix": regime_result['vix']
            },
            "horizon_days": days,
            "daily_predictions": daily_predictions,
            "summary": {
                "start_price": round(regime_result['spx_price'], 2),
                "expected_end_price": round(float(forecast['mean_path'][-1]), 2),
                "expected_return_pct": round(float(forecast['final_stats']['expected_return_pct']), 3),
                "best_case": round(float(forecast['percentiles']['p95'][-1]), 2),
                "worst_case": round(float(forecast['percentiles']['p05'][-1]), 2)
            },
            "regime_info": {
                "current_regime_avg_duration": regime_result['regime_stats']['avg_duration_days'],
                "expected_annual_return": regime_result['regime_stats']['annual_return_pct'],
                "expected_annual_vol": regime_result['regime_stats']['annual_vol_pct']
            }
        }
        
        return convert_numpy_types(response)
        
    except FileNotFoundError as e:
        raise Exception(f"Model not found. Please run nice.py first. Error: {e}")
    except Exception as e:
        raise Exception(f"Error generating short-term prediction: {e}")


def get_regime_history(days_back: int = 60) -> Dict[str, Any]:
    """
    Get historical regime classifications for the recent period.
    
    Useful for charting regime changes over time.
    """
    try:
        artifacts = get_model_artifacts()
        
        # Get regime with history
        regime_result = infer_current_regime(artifacts, return_history=True)
        
        # Get the history dataframe
        history = regime_result['history'].tail(days_back).copy()
        
        # Map regime IDs to labels
        diag = artifacts['regime_diagnostics']
        regime_map = dict(zip(diag['State'], diag['Label']))
        history['Regime_Label'] = history['Regime'].map(regime_map)
        
        # Convert to list of records
        history_records = []
        for _, row in history.iterrows():
            history_records.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "spx_price": round(float(row['SPX']), 2),
                "vix": round(float(row['VIX']), 2),
                "regime_id": int(row['Regime']),
                "regime_label": row['Regime_Label']
            })
        
        # Calculate regime distribution
        regime_counts = history['Regime_Label'].value_counts().to_dict()
        
        response = {
            "days_analyzed": days_back,
            "date_range": {
                "start": history_records[0]['date'],
                "end": history_records[-1]['date']
            },
            "current_regime": {
                "id": regime_result['regime_id'],
                "label": regime_result['regime_label']
            },
            "regime_distribution": regime_counts,
            "history": history_records
        }
        
        return convert_numpy_types(response)
        
    except Exception as e:
        raise Exception(f"Error getting regime history: {e}")


# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_all_regimes() -> Dict[str, Any]:
    """
    Get information about all possible market regimes.
    """
    try:
        artifacts = get_model_artifacts()
        diag = artifacts['regime_diagnostics']
        
        regimes = []
        for _, row in diag.iterrows():
            regimes.append({
                "state_id": int(row['State']),
                "label": row['Label'],
                "annual_return_pct": round(float(row['Ann_Ret_%']), 2),
                "annual_volatility_pct": round(float(row['Ann_Vol_%']), 2),
                "avg_duration_days": round(float(row['Avg_Duration']), 1),
                "avg_drawdown_pct": round(float(row['Avg_DD_%']), 2),
                "avg_vix": round(float(row['Avg_VIX']), 2),
                "frequency_pct": round(float(row['Pct_%']), 2)
            })
        
        response = {
            "total_regimes": len(regimes),
            "model_type": "Hidden Semi-Markov Model",
            "training_period": f"1990 to {artifacts['training_end_date']}",
            "regimes": regimes
        }
        
        return convert_numpy_types(response)
        
    except Exception as e:
        raise Exception(f"Error getting regime info: {e}")
