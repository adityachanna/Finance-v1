#  Finance v1

A comprehensive financial analysis platform combining traditional stock metrics, AI-powered insights, and advanced market regime modeling using Hidden Markov Models (HMM) and Hidden Semi-Markov Models (HSMM).

## Features

### Stock Analysis API
- **Stock Details**: Comprehensive financial metrics including PE ratio, Price-to-Book, ROE, ROA, profit margins, growth rates, beta, dividend yield, and analyst recommendations.
- **Market News**: Latest market news with AI-generated summaries providing basic understanding and company profiles.
- **Peer Comparison**: Compare stocks with industry peers on key metrics.
- **AI Explanations**: Plain English explanations of complex financial metrics and peer comparisons.

### Regime Modeling
- **HMM Models**: Hidden Markov Models for detecting market regimes (bull, bear, etc.).
- **HSMM Models**: Hidden Semi-Markov Models for more nuanced regime analysis.
- **Backtesting**: Historical backtesting of regime-based strategies.

### AI Agent
- Integrated LangChain-based AI agent for enhanced financial analysis and decision support.

## Installation

Clone the repository:

```bash
git clone https://github.com/adityachanna/Finance-v1.git
cd Finance-v1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Refer to `API_GUIDE.md` for detailed API usage examples, including how to fetch stock details, market news, peer comparisons, and perform complete analyses.

### Quick Start

```python
from details import get_complete_stock_analysis

# Get comprehensive analysis for a stock
analysis = get_complete_stock_analysis("AAPL")
print(analysis['metrics_explanation'])
```

For regime modeling, see the HMM and HSMM API files.

## Project Structure

- `details.py`: Core stock analysis functions
- `hmm_api.py`: HMM regime modeling implementation
- `hsmm_api.py`: HSMM regime modeling API
- `historical_backtest.py`: Backtesting utilities
- `combined_api.py`: Combined API endpoints
- `agent.py`: AI agent for financial insights
- `nice.py`: Main application logic
- `models/`: Directory containing trained models
- Pre-trained models: `hmm_regime_model.pkl`, `feature_scaler.pkl`
- `long_term_regime_analysis.png`: Visualization of regime analysis

## Requirements

- Python 3.8+
- Key dependencies: FastAPI, NumPy, Pandas, scikit-learn, hmmlearn, yfinance, LangChain, etc. (see `requirements.txt`)
