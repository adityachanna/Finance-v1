from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

@tool
def get_company_financials(ticker: str) -> str:
    """Useful to get financial structure, balance sheet, and company info for a given ticker."""
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
    except Exception as e:
        return f"Error fetching info for ticker {ticker}: {str(e)}"
        
    # Basic financial structure
    financials = {
        "market_cap": info.get("marketCap"),
        "total_debt": info.get("totalDebt"),
        "total_cash": info.get("totalCash"),
        "debt_to_equity": info.get("debtToEquity"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "financial_currency": info.get("financialCurrency"),
        "long_business_summary": info.get("longBusinessSummary"),
        "website": info.get("website")
    }
    
    # Try to get balance sheet
    try:
        bs = stock.balance_sheet
        if not bs.empty:
            # Take the most recent year/quarter (first column)
            # Converting to string to be safe for JSON/output
            bs_summary = bs.iloc[:, 0].to_dict() 
            # Filter out None/Nan if needed, or just cast
            financials["latest_balance_sheet"] = {str(k): str(v) for k, v in bs_summary.items()}
    except Exception as e:
        financials["balance_sheet_error"] = str(e)
        
    return str(financials)

model = ChatGroq(
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0.1,
    timeout=30
)
web_search = TavilySearchResults(max_results=10)
agent = create_agent(
    model,
    tools=[web_search, get_company_financials],
    checkpointer=InMemorySaver(),  
)

if __name__ == "__main__":
    agent.invoke(
        {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
        {"configurable": {"thread_id": "1"}},  
    )