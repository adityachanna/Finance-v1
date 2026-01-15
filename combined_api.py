import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Import modules
import details
import hmm_api
from agent import agent

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Comprehensive Financial Intelligence API",
    description="""
    ## Unified Financial API
    Combines Stock Analysis, Real-time News, Market Regime Detection (HMM), and AI Agent capabilities.
    
    ### Features:
    - **Stock Analysis**: Comprehensive fundamental analysis and peer comparison.
    - **Market News**: AI-summarized news with sentiment and company profile.
    - **Market Regime**: HMM-based bull/bear/neutral detection and forecasting.
    - **AI Agent**: Interactive financial assistant for custom queries.
    """,
    version="1.0.0"
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
    return {"status": "online", "service": "Generic Financial API", "timestamp": datetime.now().isoformat()}

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

# 2. Market Regime (from hmm_api.py)
@app.get("/api/market/{ticker}/condition", tags=["Market Regime"])
def get_market_condition(ticker: str):
    """Get current HMM-based market regime (Bull/Bear/Neutral)."""
    try:
        return hmm_api.get_market_condition(ticker)
    except Exception as e:
        logger.error(f"Error in market condition for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/{ticker}/forecast", tags=["Market Regime"])
def get_market_forecast(ticker: str, days: int = 30):
    """Get 30 or 60-day market regime forecast."""
    if days not in [30, 60]:
        raise HTTPException(status_code=400, detail="Days must be 30 or 60")
    try:
        # Use the logic from hmm_api's functionality
        # We invoke the function directly
        return hmm_api.get_regime_forecast(ticker, horizon_days=days)
    except Exception as e:
        logger.error(f"Error in forecast for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/{ticker}/forecast/short-term", tags=["Market Regime"])
def get_short_term_forecast(ticker: str):
    """
    Get a 5-day short-term prediction including price estimates.
    Returns expected price, price range, and regime sequence.
    """
    try:
        return hmm_api.get_short_term_prediction(ticker, days=5)
    except Exception as e:
        logger.error(f"Error in short-term forecast for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 3. Agent Interaction (from agent.py)
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
        
        # Extract the last message content
        last_message = result['messages'][-1]
        return {
            "response": last_message.content,
            "thread_id": request.thread_id
        }
    except Exception as e:
        logger.error(f"Error in agent query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 4. Master Combined Endpoint
@app.get("/api/dashboard/{ticker}", response_model=CombinedResponse, tags=["Dashboard"])
def get_dashboard_data(ticker: str):
    """
    Get ALL available data for a ticker in one call.
    Aggregates Analysis, News, and Market Regime.
    """
    try:
        # We can run these sequentially or ideally in parallel (using asyncio if functions were async)
        # Since underlying functions are synchronous, we run them sequentially for now.
        
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

        # 3. Market Regime
        try:
            regime = hmm_api.get_market_condition(ticker)
        except Exception as e:
            logger.warning(f"Dashboard partial failure (regime): {e}")
            regime = {"error": str(e)}

        # 4. Forecast
        try:
            forecast = hmm_api.get_regime_forecast(ticker, horizon_days=30, include_chart=False)
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
    uvicorn.run(app, host="0.0.0.0", port=8080)
