from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.chat_models import init_chat_model
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
import yfinance as yf
from dotenv import load_dotenv
import os
from langchain.agents.middleware import SummarizationMiddleware
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

# Initialize the model with OpenRouter's base URL
model = init_chat_model(

    model="xiaomi/mimo-v2-flash:free",

    model_provider="openai",

    base_url="https://openrouter.ai/api/v1",

    api_key=os.getenv("OPENROUTER_API_KEY"),


)
web_search = TavilySearchResults(max_results=6)
agent = create_agent(
    model,
    tools=[web_search, get_company_financials],
    checkpointer=InMemorySaver(),  
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 10000),
            keep=("messages", 10),
        ),
    ],
    system_prompt="""
    You are a financial assistant.

Your job is to provide clear, concise, and user-facing answers about stocks and companies using the latest available data.

Rules:
- NEVER reveal internal reasoning, analysis, or tool deliberation.
- Use tools (web_search, get_company_financials) when required.
- If multiple sources disagree, choose the most reputable and recent one silently.
- Respond in a structured, concise format.
- Avoid speculation, uncertainty narration, or source comparison in the final answer.
- If data may vary intraday, say "approximately".
- Keep responses short unless the user asks for details.
- Try to refer to latest information so call tools for getting data
Output style:
- Start with a one-line summary.
- Then list key financial metrics in bullet points.

    """)
if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Tell me about the company Apple financials."}]},
        {"configurable": {"thread_id": "1"}},
    )

    final_answer = result["messages"][-1].content
    print(final_answer)
