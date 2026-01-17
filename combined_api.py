import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Import modules
import details
import hsmm_api  # NEW: Using HSMM instead of HMM
from agent import agent
import requests

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Comprehensive Financial Intelligence API",
    description="""
    ## Unified Financial API
    Combines Stock Analysis, Real-time News, Market Regime Detection (HSMM), and AI Agent capabilities.
    
    ### Features:
    - **Stock Analysis**: Comprehensive fundamental analysis and peer comparison.
    - **Market News**: AI-summarized news with sentiment and company profile.
    - **Market Regime (HSMM)**: Hidden Semi-Markov Model based regime detection with duration modeling.
    - **Monte Carlo Forecasting**: Price forecasts using regime-aware simulations.
    - **AI Agent**: Interactive financial assistant for custom queries.
    
    ### HSMM Model Info:
    The market regime model uses a 9-state Hidden Semi-Markov Model trained on S&P 500 data from 1990-2022.
    Regimes include: Strong Bull, Bull Market, Mild Bull, Recovery, Sideways, High Volatility, Correction, Bear Market, Crisis.
    """,
    version="2.0.0"
)

# CORS Config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class AgentQuery(BaseModel):
    query: str
    thread_id: str = "default_thread"

class CombinedResponse(BaseModel):
    ticker: str
    timestamp: str
    stock_analysis: Optional[Dict[str, Any]] = None
    market_news: Optional[Dict[str, Any]] = None
    market_regime: Optional[Dict[str, Any]] = None
    regime_forecast: Optional[Dict[str, Any]] = None

# --- Endpoints ---

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "online", "service": "Financial Intelligence API v2.0 (HSMM)", "timestamp": datetime.now().isoformat()}

# 1. Stock Details & Analysis (from details.py)
@app.get("/api/stock/{ticker}/analysis", tags=["Stock Analysis"])
def get_stock_analysis(ticker: str):
    """Get complete stock analysis including fundamentals, peers, and AI insights."""
    try:
        return details.get_complete_stock_analysis(ticker)
    except Exception as e:
        logger.error(f"Error in stock analysis for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{ticker}/news", tags=["Stock Analysis"])
def get_stock_news(ticker: str, max_items: int = 5):
    """Get AI-summarized market news with analysis."""
    try:
        return details.get_market_news(ticker, max_news=max_items)
    except Exception as e:
        logger.error(f"Error in news for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# 2. HSMM Market Regime Endpoints (NEW)
# ============================================

@app.get("/api/market/condition", tags=["Market Regime (HSMM)"])
def get_market_condition():
    """
    Get current HSMM-based market regime.
    
    The model analyzes S&P 500 to determine the overall market state.
    Returns one of 9 possible regimes: Strong Bull, Bull Market, Mild Bull, 
    Recovery, Sideways, High Volatility, Correction, Bear Market, or Crisis.
    
    Includes:
    - Current regime ID and label
    - S&P 500 price and VIX level
    - Historical characteristics of this regime (avg return, volatility, duration)
    - Model metadata
    """
    try:
        return hsmm_api.get_market_condition()
    except Exception as e:
        logger.error(f"Error in market condition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/forecast", tags=["Market Regime (HSMM)"])
def get_market_forecast(
    days: int = Query(default=60, ge=5, le=120, description="Forecast horizon in days"),
    simulations: int = Query(default=2000, ge=100, le=10000, description="Number of Monte Carlo simulations"),
    include_paths: bool = Query(default=False, description="Include sample price paths for charting")
):
    """
    Get HSMM-based Monte Carlo price forecast.
    
    Uses the current market regime and Semi-Markov transition dynamics to 
    simulate thousands of possible future price paths.
    
    Returns:
    - Expected price and return
    - Bull case (95th percentile) and Bear case (5th percentile)
    - Confidence intervals (50% and 90%)
    - Optionally: Mean path and sample paths for visualization
    """
    try:
        return hsmm_api.get_regime_forecast(
            horizon_days=days,
            n_simulations=simulations,
            include_paths=include_paths
        )
    except Exception as e:
        logger.error(f"Error in forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/forecast/short-term", tags=["Market Regime (HSMM)"])
def get_short_term_forecast(
    days: int = Query(default=5, ge=1, le=10, description="Prediction horizon (1-10 days)")
):
    """
    Get short-term price prediction with daily estimates.
    
    Useful for near-term trading decisions. Returns:
    - Daily expected price
    - Price range (25th-75th percentile)
    - Overall expected return
    - Best/worst case scenarios
    """
    try:
        return hsmm_api.get_short_term_prediction(days=days)
    except Exception as e:
        logger.error(f"Error in short-term forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/regimes", tags=["Market Regime (HSMM)"])
def get_all_regimes():
    """
    Get information about all possible market regimes.
    
    Returns detailed statistics for each of the 9 regimes including:
    - Expected annual return and volatility
    - Average duration in days
    - Historical frequency
    - Average VIX level
    """
    try:
        return hsmm_api.get_all_regimes()
    except Exception as e:
        logger.error(f"Error getting regimes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/history", tags=["Market Regime (HSMM)"])
def get_regime_history(
    days: int = Query(default=60, ge=10, le=750, description="Number of days of history (max 750 = ~3 years)")
):
    """
    Get historical regime classifications.
    
    Returns daily regime data for charting regime transitions over time.
    Includes date, price, VIX, and regime classification for each day.
    """
    try:
        return hsmm_api.get_regime_history(days_back=days)
    except Exception as e:
        logger.error(f"Error getting regime history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# 3. Agent Interaction (from agent.py)
# ============================================

@app.post("/api/agent/query", tags=["AI Agent"])
def query_agent(request: AgentQuery):
    """
    Ask the AI Agent a question.
    The agent uses LangChain with access to web search and financial tools.
    """
    try:
        input_message = {"messages": [{"role": "user", "content": request.query}]}
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Invoke agent
        result = agent.invoke(input_message, config)
        response=result["messages"][-1].content
        # Extract the last message conten
        return {
            "response": response,
            "thread_id": request.thread_id
        }
    except Exception as e:
        logger.error(f"Error in agent query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 3.5 Symbol Search
@app.get("/api/search", tags=["Utility"])
def search_ticker(q: str = Query(..., min_length=1)):
    """
    Search for a ticker symbol by name or keyword.
    Returns a list of matching symbols with metadata.
    """
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": q, "quotesCount": 10, "newsCount": 0}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        resp = requests.get(url, params=params, headers=headers)
        data = resp.json()
        
        results = []
        if "quotes" in data:
            for quote in data["quotes"]:
                 results.append({
                     "symbol": quote.get("symbol"),
                     "name": quote.get("shortname") or quote.get("longname"),
                     "exch": quote.get("exchange"),
                     "type": quote.get("quoteType")
                 })
        return {"count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# 4. Master Combined Endpoint
# ============================================

@app.get("/api/dashboard/{ticker}", response_model=CombinedResponse, tags=["Dashboard"])
def get_dashboard_data(ticker: str):
    """
    Get ALL available data for a ticker in one call.
    Aggregates Analysis, News, and Market Regime (HSMM).
    
    Note: Market regime is based on S&P 500 (SPX) regardless of ticker queried,
    as it represents overall market conditions.
    """
    try:
        # 1. Stock Analysis
        try:
            analysis = details.get_complete_stock_analysis(ticker)
        except Exception as e:
            logger.warning(f"Dashboard partial failure (analysis): {e}")
            analysis = {"error": str(e)}

        # 2. News
        try:
            news = details.get_market_news(ticker, max_news=5)
        except Exception as e:
            logger.warning(f"Dashboard partial failure (news): {e}")
            news = {"error": str(e)}

        # 3. Market Regime (HSMM)
        try:
            regime = hsmm_api.get_market_condition()
        except Exception as e:
            logger.warning(f"Dashboard partial failure (regime): {e}")
            regime = {"error": str(e)}

        # 4. Forecast (HSMM Monte Carlo)
        try:
            forecast = hsmm_api.get_regime_forecast(horizon_days=30, n_simulations=1000)
        except Exception as e:
             logger.warning(f"Dashboard partial failure (forecast): {e}")
             forecast = {"error": str(e)}

        return {
            "ticker": ticker.upper(),
            "timestamp": datetime.now().isoformat(),
            "stock_analysis": analysis,
            "market_news": news,
            "market_regime": regime,
            "regime_forecast": forecast
        }

    except Exception as e:
        logger.error(f"Critical error in dashboard for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
