# AlphaSharp: Financial Intelligence API Platform

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688.svg)

**AlphaSharp** is a sophisticated Financial Intelligence API platform that integrates real-time stock analysis, market news aggregation, advanced market regime detection using Hidden Semi-Markov Models (HSMM), and AI-powered financial assistance. 

Designed for institutional-grade financial intelligence, AlphaSharp provides a unified interface for developers and analysts to consume complex market insights through simple RESTful endpoints.

---

## 🚀 Key Features

### 1. Market Regime Detection (HSMM)
Utilizes a pre-trained **Hidden Semi-Markov Model (HSMM)** trained on over 30 years of S&P 500 data (1990-2022) to classify market states into 9 distinct regimes:
- 🟢 **Bullish**: Strong Bull, Bull Market, Mild Bull, Recovery
- 🟡 **Neutral**: Sideways, High Volatility
- 🔴 **Bearish**: Correction, Bear Market, Crisis

### 2. Monte Carlo Price Forecasting
Generates probabilistic price forecasts using regime-aware transition dynamics. Unlike simple linear projections, our model samples from learned regime durations and state-specific return distributions to produce:
- 60-day price trajectories with confidence intervals (50%, 90%)
- Best-case (95th percentile) and Worst-case (5th percentile) scenarios
- Short-term (1-10 day) trading-focused predictions

### 3. Comprehensive Stock Analysis
Deep-dive into individual company performance:
- **Fundamentals**: Key valuation, profitability, and growth metrics
- **Peer Comparison**: Dynamic industry benchmarking
- **AI Explanations**: Plain-English interpretation of complex financial data using LLMs
- **Sentiment Analysis**: AI-summarized recent news and market sentiment

### 4. Interactive AI Financial Assistant (Agent)
An advanced agentic assistant that can:
- **Real-time Research**: Perform live web searches for the latest financial news via Tavily.
- **Deep Financials**: Retrieve company-specific Balance Sheets, Income Statements, and key ratios.
- **Contextual Memory**: Maintain conversational memory across multiple queries using `thread_id`.
- **Intelligent Summarization**: Automatically summarizes long conversations to stay within token limits.

---

## 🛠 Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Machine Learning**: `hmmlearn`, `scikit-learn`, `joblib`, `numpy`, `pandas`
- **AI/Agents**: LangChain, LangGraph, Groq, OpenRouter (Xiaomi Mimo v2 Flash)
- **Data Source**: Yahoo Finance (yfinance), Tavily Search
- **Deployment**: Ready for Docker/Gunicorn

---

## 🚦 Getting Started

### Prerequisites
- Python 3.8 or higher
- API Keys for Groq, OpenRouter, and Tavily

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adityachanna/Finance-v1.git
   cd Finance-v1
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

5. **Run the API Server:**
   ```bash
   python combined_api.py
   ```
   The API will be available at `http://localhost:4000`.

---

## 📖 API Documentation

AlphaSharp provides interactive documentation via Swagger UI. Once the server is running, visit:
- **Swagger UI**: `http://localhost:4000/docs`
- **ReDoc**: `http://localhost:4000/redoc`

### 1. Unified Dashboard
`GET /api/dashboard/{ticker}`

Returns a complete intelligence profile for a stock ticker.
- **Aggregated Data**: Fundamentals, AI-summarized News, Peer Benchmarking, and Current Market Regime.
- **Fail-Safe**: If any sub-module (e.g., news) fails, the endpoint returns a `200 OK` with remaining data and an `error` field in the failed key.

**Sample Usage:**
`curl http://localhost:4000/api/dashboard/NVDA`

---

### 2. Market Regime (HSMM)
`GET /api/market/condition`

Retrieves the current state of the market based on the S&P 500.

**Response Fields:**
- `regime_id`: Index (0-8) of the detected regime.
- `label`: Human-readable description (e.g., "Crisis", "Strong Bull").
- `characteristics`: Statistics including `avg_duration_days` and `expected_annual_return`.

---

### 3. Price Forecasting (Monte Carlo)
- **Standard Forecast** (`GET /api/market/forecast`): Horizon 5-120 days. Returns confidence intervals (50%, 90%) and bull/bear cases.
- **Short-Term Prediction** (`GET /api/market/forecast/short-term`): Precision 1-10 day horizon with daily price estimates and confidence ranges.

**Parameters:**
- `days`: Horizon (default: 60 for standard, 5 for short-term).
- `simulations`: Number of runs (100-10,000).
- `include_paths`: Boolean to include raw path data for charting.

---

### 4. Financial AI Agent
`POST /api/agent/query`

Submit a financial query to the AI assistant.

**Request Body:**
```json
{
  "query": "What is the current sentiment for Nvidia and how do their margins compare to peers?",
  "thread_id": "optional-session-id"
}
```

**Response:**
```json
{
  "response": "Based on recent earnings and market data, NVIDIA (NVDA) maintains a dominant market position...",
  "thread_id": "optional-session-id"
}
```

---

### 5. Symbol Search & History
- **Fuzzy Search** (`GET /api/search?q={query}`): Map natural language company names to tickers using Yahoo Finance's search API.
- **Regime History** (`GET /api/market/history?days=60`): Historical daily regime classifications for the S&P 500 (up to 750 days).
- **All Regimes** (`GET /api/market/regimes`): Static definitions and historical stats for all 9 modeled market states.

---

### Error Handling
The API returns standard HTTP status codes.
- `200 OK`: Successful request.
- `422 Unprocessable Entity`: Validation failure (e.g., invalid ticker).
- `500 Internal Server Error`: Server-side issues or external API failures.

Dashboard partial failures: If one module (e.g. News) fails, the dashboard endpoint still returns the remaining data with an `error` field in the failed module's key.

---

## 🏗 Core Components & File Breakdown

### 📂 `combined_api.py` (Main Entry Point)
The central orchestration layer of AlphaSharp. It exposes a unified FastAPI interface that integrates all sub-modules.
- **Dashboard Aggregator (`/api/dashboard/{ticker}`)**: The master endpoint that concurrently fetches stock fundamentals, AI-summarized news, and current global market regime data.
- **Safety & Error Handling**: Implements partial failure logic—if one service (like News) is down, the API still returns available data (like Fundamentals) with an error flag.
- **CORS Enabled**: Pre-configured for seamless integration with modern frontend frameworks (React, Vue, Next.js).

### 📂 `details.py` (Stock Analysis Engine)
Handles the retrieval and interpretation of company-specific data.
- **Dual-Stage AI Processing**:
    1. **News Summary**: Processes raw RSS/News feeds into a factual event list.
    2. **Strategic Analysis**: Combines news events with hard financial metrics to produce a "Verdict" for investors.
- **Fundamental Aggregator**: Fetches over 40+ data points including Profitability (ROE/ROA), Valuation (P/E, PEG), and Financial Health (Debt-to-Equity).
- **Peer Comparison**: Dynamically identifies industry competitors and performs side-by-side benchmarking.

### 📂 `hsmm_inference.py` (ML Inference Engine)
The mathematical core that runs the Hidden Semi-Markov Model.
- **Live Feature Engineering**: Real-time calculation of a 9-dimensional feature vector, including:
    - **Volatility**: Annualized Realized Vol and VIX-to-MA50 ratios.
    - **Momentum**: Multi-timeframe returns for SPX (S&P 500) and NDX (Nasdaq 100).
    - **Structure**: Real-time Correlation and Beta calculation between asset classes.
- **Monte Carlo Simulator**: Generates 2,000+ probabilistic price paths by sampling from learned regime duration distributions and state-specific return volatilities.

### 📂 `nice.py` (HSMM Training & Research)
The research-focused script used to build and validate the market model.
- **GaussianHSMM Class**: A custom wrapper around `hmmlearn` that introduces explicit state duration modeling (Gamma Distribution), resolving the "memoryless" limitation of standard HMMs.
- **Model Optimization**: Uses Bayesian Information Criterion (BIC) to automatically find the optimal number of market states (currently 9).
- **Diagnostic Suite**: Generates regime-specific statistics (Avg Duration, Expected Annual Return) and historical backtest visualizations.

### 📂 `agent.py` (AI Assistant Logic)
Configures the "Agentic" capabilities of the platform.
- **Tool Integration**: Connects the LLM (Xiaomi Mimo v2 Flash) to the real world via:
    - `web_search`: Live search powered by Tavily.
    - `get_company_financials`: A custom tool that extracts deep balance sheet data using `yfinance`.
- **Memory Management**: Uses `InMemorySaver` for thread-persistent conversations and `SummarizationMiddleware` to handle long-running dialogues.

### 📂 `requirements.txt` (Dependencies)
- **Web**: `fastapi`, `uvicorn`
- **Data/ML**: `numpy`, `pandas`, `scikit-learn`, `hmmlearn`
- **AI**: `langchain`, `langgraph`, `tavily-python`
- **Finance**: `yfinance`

---

## 📈 Model Information

The **Hidden Semi-Markov Model (HSMM)** differs from standard HMMs by explicitly modeling the **duration** a market stays in a particular state. This allows for far more realistic simulations of market cycles compared to memoryless transitions. Our model uses 9 engineered features, including realized volatility, VIX levels, Bollinger widths, and log returns.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contact

**Project Collaborators** : Aditya Channa, Madhav Kapoor
**Repository**: [Finance-v1](https://github.com/adityachanna/Finance-v1)

