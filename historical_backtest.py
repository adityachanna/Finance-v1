import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import hmm_api  # Import your existing API model

def run_long_term_backtest(ticker="^GSPC", start_date="1985-01-01"):
    print(f"🔄 Fetching data for {ticker} from {start_date} to present...")
    
    # 1. Fetch Long-Term Data
    df = yf.download(ticker, start=start_date, progress=True, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.reset_index()
        # Handle 'Ticker' level if present, or just flatten
        try:
            df.columns = df.columns.get_level_values(0) 
        except:
            pass
            
    # Normalize columns
    df = df.rename(columns={"Date": "Date", "Close": "Close", "Adj Close": "Close"})
    df = df[["Date", "Close"]].dropna()
    
    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    
    print(f"✅ Data loaded: {len(df)} trading days.")

    # 2. Run HMM Model 
    print("🧠 Running HMM Inference on historical data...")
    detector = hmm_api.HMMRegimeDetector()
    
    # valid_df will have the computed features (Log_Return, Volatility)
    # and the inferred regimes
    results_df = detector.get_historical_regimes(df)
    
    # 3. Calculate Performance Stats
    print("📊 Calculating regime statistics...")
    stats = results_df.groupby("regime_name").agg(
        Count=("Date", "count"),
        Avg_Return=("Log_Return", "mean"),
        Avg_Volatility=("Volatility", "mean")
    )
    stats["Percent_Time"] = (stats["Count"] / len(results_df)) * 100
    stats["Annualized_Return"] = stats["Avg_Return"] * 252 * 100
    
    print("\n" + "="*50)
    print(f"REGIME ANALYSIS (1985 - Present)")
    print("="*50)
    print(stats[["Percent_Time", "Annualized_Return", "Avg_Volatility"]])
    print("="*50 + "\n")

    # 4. Generate Visualization
    print("🎨 Generating chart...")
    plot_historical_regimes(results_df, ticker)

def plot_historical_regimes(df, ticker):
    # Set style
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1], sharex=True)
    fig.patch.set_facecolor('#0f172a')
    
    dates = df["Date"]
    price = df["Close"]
    regimes = df["regime_state"]
    volatility = df["Volatility"]
    
    # --- TOP PLOT: PRICE & REGIME BACKGROUND ---
    ax1.set_facecolor('#1e293b')
    ax1.semilogy(dates, price, color='white', linewidth=1, label=f'{ticker} Price (Log Scale)')
    
    # Create colored spans
    # We identify chunks where the regime stays the same to avoid 10,000 separate polygon calls
    # This speeds up plotting significantly
    
    colors = {
        0: '#ef4444', # Bear - Red
        1: '#22c55e', # Bull - Green
        2: '#eab308'  # Neutral - Yellow
    }
    
    labels_added = set()
    
    # Efficiently plot spans
    # Group consecutive regimes
    df['block'] = (df['regime_state'] != df['regime_state'].shift()).cumsum()
    blocks = df.groupby('block')
    
    for _, block in blocks:
        start_date = block['Date'].iloc[0]
        end_date = block['Date'].iloc[-1]
        regime = block['regime_state'].iloc[0]
        regime_eval = hmm_api.REGIME_NAMES[regime]
        
        # Add label only once for legend
        label = regime_eval if regime_eval not in labels_added else None
        if label: labels_added.add(regime_eval)
        
        ax1.axvspan(start_date, end_date, color=colors[regime], alpha=0.3, label=label)

    ax1.set_title(f"{ticker} Long-Term Market Regimes (1985 - Present)", fontsize=18, color='white', pad=15)
    ax1.set_ylabel("Price (Log Scale)", fontsize=12, color='white')
    ax1.grid(True, which='both', linestyle='--', alpha=0.1)
    ax1.legend(loc='upper left', fontsize=10, facecolor='#1e293b')

    # --- BOTTOM PLOT: INDICATOR (VOLATILITY) ---
    ax2.set_facecolor('#1e293b')
    ax2.plot(dates, volatility, color='#38bdf8', linewidth=0.8, label='Market Volatility (Model Feature)')
    
    # Add threshold line for high volatility context (approximate)
    avg_vol = volatility.mean()
    ax2.axhline(avg_vol, color='white', linestyle='--', alpha=0.5, label=f'Avg Volatility ({avg_vol:.1%})')
    
    ax2.set_title("Market Volatility (Key Model Indicator)", fontsize=14, color='white')
    ax2.set_ylabel("Annualized Volatility", fontsize=12, color='white')
    ax2.set_xlabel("Year", fontsize=12, color='white')
    ax2.grid(True, linestyle='--', alpha=0.1)
    ax2.legend(loc='upper left', fontsize=10, facecolor='#1e293b')
    
    # Formatting
    plt.tight_layout()
    
    # Save chart
    filename = "long_term_regime_analysis.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    print(f"✨ Chart saved successfully to: {filename}")
    
if __name__ == "__main__":
    run_long_term_backtest()
