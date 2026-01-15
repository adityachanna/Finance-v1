"""
Stock Details API with AI-Powered Explanations
Provides comprehensive stock metrics and easy-to-understand explanations with peer comparison
"""

import yfinance as yf
from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize AI model for explanations
model = ChatGroq(
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0.3,
    timeout=60
)


def get_stock_details(ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive stock details from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        Dictionary containing key financial metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get historical data for YoY growth calculation
        hist = stock.history(period="2y")
        
        # Calculate YoY price growth if we have enough data
        yoy_growth = None
        if len(hist) >= 252:  # Approximately 1 year of trading days
            current_price = hist['Close'].iloc[-1]
            year_ago_price = hist['Close'].iloc[-252]
            yoy_growth = ((current_price - year_ago_price) / year_ago_price) * 100
        
        # Extract key metrics
        details = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            
            # Price metrics
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "currency": info.get("currency", "USD"),
            "day_change_percent": info.get("regularMarketChangePercent"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            
            # Valuation metrics
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "pe_ratio_trailing": info.get("trailingPE"),
            "pe_ratio_forward": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "peg_ratio": info.get("pegRatio"),
            
            # Profitability metrics
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),  # Return on Equity
            "roa": info.get("returnOnAssets"),  # Return on Assets
            
            # Growth metrics
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "yoy_price_growth": yoy_growth,
            
            # Per share metrics
            "eps_trailing": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
            
            # Dividend metrics
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            
            # Risk metrics
            "beta": info.get("beta"),
            
            # Volume
            "avg_volume": info.get("averageVolume"),
            "volume": info.get("volume"),
            
            # Analyst recommendations
            "recommendation": info.get("recommendationKey"),
            "target_mean_price": info.get("targetMeanPrice"),
            "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
        }
        
        # Clean up None values and format percentages
        for key, value in details.items():
            if value is None:
                details[key] = "N/A"
            elif isinstance(value, float):
                # Format percentages
                if key in ["profit_margin", "operating_margin", "roe", "roa", "revenue_growth", 
                          "earnings_growth", "dividend_yield", "day_change_percent"]:
                    details[key] = round(value * 100, 2) if value != "N/A" else "N/A"
                else:
                    details[key] = round(value, 2)
        
        return details
        
    except Exception as e:
        return {"error": f"Failed to fetch stock details for {ticker}: {str(e)}"}


def get_peer_comparison(ticker: str, peers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compare stock with industry peers.
    
    Args:
        ticker: Stock ticker symbol
        peers: Optional list of peer tickers. If None, will fetch from yfinance
        
    Returns:
        Dictionary containing comparison data
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get peers if not provided
        if peers is None:
            # Try to get recommendations/similar stocks
            try:
                recommendations = stock.recommendations
                peers = []
                # yfinance doesn't have a direct "peers" endpoint, so we'll use common peers
                # For demonstration, we'll use predefined peer groups for major stocks
                info = stock.info
                sector = info.get("sector", "")
                
                # Define some common peer groups (you can expand this)
                peer_groups = {
                    "AAPL": ["MSFT", "GOOGL", "META"],
                    "MSFT": ["AAPL", "GOOGL", "AMZN"],
                    "GOOGL": ["MSFT", "META", "AMZN"],
                    "TSLA": ["F", "GM", "RIVN"],
                    # Add more as needed
                }
                
                ticker_upper = ticker.upper()
                if ticker_upper in peer_groups:
                    peers = peer_groups[ticker_upper]
                else:
                    # Default to some tech giants for comparison
                    peers = ["AAPL", "MSFT", "GOOGL"][:2]  # Limit to 2 peers
            except:
                peers = []
        
        # Get details for main stock
        main_stock = get_stock_details(ticker)
        
        # Get details for peers
        peer_data = {}
        for peer in peers[:3]:  # Limit to 3 peers
            if peer.upper() != ticker.upper():
                peer_data[peer] = get_stock_details(peer)
        
        # Create comparison
        comparison = {
            "ticker": ticker.upper(),
            "main_stock": main_stock,
            "peers": peer_data,
            "comparison_metrics": {}
        }
        
        # Compare key metrics
        if "error" not in main_stock:
            metrics_to_compare = [
                "pe_ratio_trailing", "market_cap", "profit_margin", 
                "roe", "revenue_growth", "beta", "dividend_yield"
            ]
            
            for metric in metrics_to_compare:
                comparison["comparison_metrics"][metric] = {
                    ticker.upper(): main_stock.get(metric, "N/A")
                }
                for peer, peer_info in peer_data.items():
                    if "error" not in peer_info:
                        comparison["comparison_metrics"][metric][peer.upper()] = peer_info.get(metric, "N/A")
        
        return comparison
        
    except Exception as e:
        return {"error": f"Failed to compare with peers: {str(e)}"}


def explain_stock_metrics(stock_details: Dict[str, Any]) -> str:
    """
    Use AI to explain stock metrics in simple, consumer-friendly language.
    
    Args:
        stock_details: Dictionary from get_stock_details()
        
    Returns:
        String containing easy-to-understand explanations
    """
    if "error" in stock_details:
        return f"Unable to explain metrics: {stock_details['error']}"
    
    prompt = f"""You are a friendly financial advisor explaining stock metrics to a beginner investor.

Here are the stock details for {stock_details.get('company_name')} ({stock_details.get('ticker')}):

{json.dumps(stock_details, indent=2)}

Please explain the following in simple, easy-to-understand terms (as if talking to someone with no finance background):

1. **What is this company?** (Sector, industry, what they do)
2. **Stock Price Summary**: Current price, 52-week range, and what this tells us
3. **Is it expensive or cheap?** (Explain PE ratio, Price-to-Book in simple terms)
4. **How profitable is it?** (Explain profit margins, ROE in simple language)
5. **Is it growing?** (Explain revenue and earnings growth)
6. **How risky is it?** (Explain beta in simple terms)
7. **Does it pay dividends?** (If yes, explain dividend yield)
8. **Overall Assessment**: A simple summary - is this typically considered a value stock, growth stock, stable dividend stock, etc.?

Use simple analogies where helpful. Avoid jargon. If a metric is "N/A", just skip it.
Keep the total explanation concise but informative (around 400-500 words).
"""
    
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def explain_peer_comparison(comparison_data: Dict[str, Any]) -> str:
    """
    Use AI to explain how the stock compares to its peers in simple language.
    
    Args:
        comparison_data: Dictionary from get_peer_comparison()
        
    Returns:
        String containing easy-to-understand comparison explanation
    """
    if "error" in comparison_data:
        return f"Unable to explain comparison: {comparison_data['error']}"
    
    prompt = f"""You are a friendly financial advisor comparing stocks for a beginner investor.

Here is the comparison data:
Main Stock: {comparison_data.get('ticker')}

{json.dumps(comparison_data.get('comparison_metrics', {}), indent=2)}

Please explain in simple terms:

1. **How does this stock compare to its peers?**
   - Is it more expensive or cheaper based on PE ratio?
   - Is it more or less profitable (profit margin, ROE)?
   - Is it growing faster or slower (revenue growth)?
   - Is it more or less risky (beta)?

2. **What does this mean for investors?**
   - What are the relative strengths of this stock?
   - What are the relative weaknesses?

3. **Simple verdict**: In 1-2 sentences, summarize whether this stock stands out positively or negatively compared to peers.

Keep it simple and conversational. Use comparisons like "higher than", "lower than", "similar to".
Keep the explanation concise (around 250-300 words).
If any data is "N/A", just skip that comparison.
"""
    
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating comparison explanation: {str(e)}"


def get_market_news(ticker: str, max_news: int = 10) -> Dict[str, Any]:
    """
    Fetch market news and generate AI summaries using two separate calls.
    1. News Summary: Based on recent headlines.
    2. Analysis Summary: Based on News Summary + Stock Info.
    Returns full stock details in the response.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Fetch News
        try:
            news = stock.news
        except Exception as e:
            return {"error": f"Failed to fetch news: {str(e)}", "news_data": [], "summary": {}}

        # 2. Process News Data
        news_data = []
        if news:
            for article in news[:max_news]:
                content = article.get('content', article) 
                
                def get_val(d, key, default="N/A"):
                    return d.get(key, default)

                title = get_val(content, 'title')
                publisher = "Yahoo Finance" 
                if 'provider' in article:
                     publisher = article['provider'].get('displayName', 'Yahoo Finance')
                elif 'provider' in content:
                     publisher = content['provider'].get('displayName', 'Yahoo Finance')
                     
                link = "N/A"
                if 'clickThroughUrl' in article and article['clickThroughUrl']:
                    link = article['clickThroughUrl'].get('url', 'N/A')
                elif 'canonicalUrl' in article and article['canonicalUrl']:
                     link = article['canonicalUrl'].get('url', 'N/A')
                elif 'clickThroughUrl' in content and content['clickThroughUrl']:
                     link = content['clickThroughUrl'].get('url', 'N/A')
                elif 'canonicalUrl' in content and content['canonicalUrl']:
                     link = content['canonicalUrl'].get('url', 'N/A')

                news_item = {
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "publish_time": get_val(content, 'pubDate'),
                    "summary": get_val(content, 'summary'),
                    "thumbnail": "N/A"
                }
                
                # Try to get a thumbnail
                thumb_data = content.get('thumbnail', article.get('thumbnail'))
                if thumb_data and 'resolutions' in thumb_data:
                     if len(thumb_data['resolutions']) > 0:
                          news_item["thumbnail"] = thumb_data['resolutions'][0].get('url', 'N/A')
                          
                news_data.append(news_item)

        # 3. Get Stock Info (for context and return)
        try:
             stock_info = stock.info
             
             # Process key factors for the user as requested
             # Extracting a comprehensive set of important keys for the end user
             
             def fmt(val, is_pct=False, prefix="", suffix=""):
                 if val is None or val == "N/A": return "N/A"
                 try:
                     if is_pct: return f"{val * 100:.2f}%"
                     if isinstance(val, (int, float)): 
                         # Format large numbers
                         if val > 1e12: return f"{prefix}{val/1e12:.2f}T{suffix}"
                         if val > 1e9: return f"{prefix}{val/1e9:.2f}B{suffix}"
                         if val > 1e6: return f"{prefix}{val/1e6:.2f}M{suffix}"
                         return f"{prefix}{val:.2f}{suffix}"
                     return f"{prefix}{val}{suffix}"
                 except: return str(val)

             key_data = {
                 "company_profile": {
                     "summary": stock_info.get("longBusinessSummary", "N/A"),
                     "sector": stock_info.get("sector", "N/A"),
                     "industry": stock_info.get("industry", "N/A"),
                     "employees": stock_info.get("fullTimeEmployees", "N/A"),
                     "website": stock_info.get("website", "N/A"),
                     "location": ", ".join([stock_info.get(k, "") for k in ['city', 'state', 'country'] if stock_info.get(k)])
                 },
                 "market_data": {
                     "current_price": fmt(stock_info.get("currentPrice"), prefix="$"),
                     "market_cap": fmt(stock_info.get("marketCap"), prefix="$"),
                     "volume": fmt(stock_info.get("volume")),
                     "avg_volume": fmt(stock_info.get("averageVolume")),
                     "fifty_two_week_range": f"{fmt(stock_info.get('fiftyTwoWeekLow'), prefix='$')} - {fmt(stock_info.get('fiftyTwoWeekHigh'), prefix='$')}"
                 },
                 "valuation_metrics": {
                     "trailing_pe": fmt(stock_info.get("trailingPE")),
                     "forward_pe": fmt(stock_info.get("forwardPE")),
                     "peg_ratio": fmt(stock_info.get("pegRatio")),
                     "price_to_book": fmt(stock_info.get("priceToBook")),
                     "price_to_sales": fmt(stock_info.get("priceToSalesTrailing12Months"))
                 },
                 "financial_health": {
                     "total_cash": fmt(stock_info.get("totalCash"), prefix="$"),
                     "total_debt": fmt(stock_info.get("totalDebt"), prefix="$"),
                     "debt_to_equity": fmt(stock_info.get("debtToEquity")),
                     "free_cash_flow": fmt(stock_info.get("freeCashflow"), prefix="$"),
                     "current_ratio": fmt(stock_info.get("currentRatio"))
                 },
                 "profitability_growth": {
                     "profit_margin": fmt(stock_info.get("profitMargins"), is_pct=True),
                     "operating_margin": fmt(stock_info.get("operatingMargins"), is_pct=True),
                     "return_on_equity": fmt(stock_info.get("returnOnEquity"), is_pct=True),
                     "revenue_growth": fmt(stock_info.get("revenueGrowth"), is_pct=True),
                     "earnings_growth": fmt(stock_info.get("earningsGrowth"), is_pct=True)
                 },
                 "analyst_sentiment": {
                     "recommendation": stock_info.get("recommendationKey", "N/A").upper(),
                     "target_price": fmt(stock_info.get("targetMeanPrice"), prefix="$"),
                     "number_of_analysts": stock_info.get("numberOfAnalystOpinions", "N/A")
                 }
             }

             # Filter for key metrics to keep context manageable but informative for AI
             relevant_info = {
                 k: v for k, v in stock_info.items() 
                 if k in ['longName', 'sector', 'industry', 'marketCap', 'trailingPE', 'forwardPE', 
                          'beta', 'profitMargins', 'revenueGrowth', 'targetMeanPrice', 'currentPrice', 
                          'totalCash', 'totalDebt']
             }
             company_name = stock_info.get('longName', ticker.upper())
        except Exception as e:
             print(f"Error fetching stock info: {e}")
             stock_info = {}
             relevant_info = {}
             key_data = {}
             company_name = ticker.upper()

        # 4. AI Call 1: News Summary
        print(f"Generating AI News Summary for {ticker}...")
        
        if news_data:
            news_context = "\n".join([f"- {item['title']}: {item['summary']}" for item in news_data[:7]])
            
            news_prompt = f"""Summarize the recent news for {company_name} ({ticker}).
            
Headlines:
{news_context}

Provide a **NEWS SUMMARY** (5-10 lines):
- Focus strictly on the facts and events reported.
- DO NOT provide analysis or opinions yet.
- Just list what is happening.
"""
            try:
                news_summary = model.invoke(news_prompt).content.strip()
            except Exception as e:
                news_summary = f"Error generating news summary: {str(e)}"
        else:
            news_summary = "No recent news found for this stock."

        # 5. AI Call 2: Analysis Summary (News Summary + Stock Info)
        print(f"Generating AI Analysis Summary for {ticker}...")
        
        analysis_prompt = f"""You are a financial analyst. Analyze {company_name} ({ticker}) for a beginner investor.

**Context:**
1. **Company Stats**: {json.dumps(relevant_info, indent=2)}
2. **Review of Recent News**:
{news_summary}

**Task: Provide an ANALYSIS SUMMARY**:
- Start with a clear **Company Profile**: What do they do? (Use the stats/summary provided).
- **Valuation & Standing**: How do the stats (PE: {relevant_info.get('trailingPE', 'N/A')}, Beta: {relevant_info.get('beta', 'N/A')}) match the current news sentiment?
- **Verdict/Insight**: Explain what this means for a non-expert user. Is the company stable? Growing? Risky? Connect the numbers with the news.

Keep it simple, conversational, and direct.
"""
        try:
            analysis_summary = model.invoke(analysis_prompt).content.strip()
        except Exception as e:
            analysis_summary = f"Error generating analysis: {str(e)}"

        return {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "total_news_count": len(news_data),
            "key_data": key_data,  # The requested semi-presentable data
            "news_data": news_data,
            "summary": {
                "news_summary": news_summary,
                "analysis_summary": analysis_summary
            }
        }

    except Exception as e:
        return {"error": f"Unexpected error in market news: {str(e)}"}

def explain_stock_metrics(stock_details: Dict[str, Any]) -> str:
    """
    Generate a purely analytical AI response for the stock.
    NO educational definitions. Focus on the company's numbers.
    """
    if "error" in stock_details:
        return f"Unable to analyze: {stock_details['error']}"
    
    # Prompt explicitly forbids defining terms
    prompt = f"""Analyze the provided financial metrics for {stock_details.get('company_name')} ({stock_details.get('ticker')}).
    
Metrics:
{json.dumps(stock_details, indent=2)}

Task: Provide a strategic financial analysis.
- Focus ONLY on interpreting the data for this specific company.
- DO NOT define what "PE Ratio", "Beta", or "ROE" means. The user knows this.
- Analyze:
  1. Valuation (Cheap/Expensive?)
  2. Growth & Profitability Health
  3. Risk Profile
  4. Analyst Sentiment
- Conclude with a brief verdict on the company's financial position.
"""
    
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"


def get_complete_stock_analysis(ticker: str, peers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Master function that gets stock details, peer comparison, and AI explanations.
    
    Args:
        ticker: Stock ticker symbol
        peers: Optional list of peer tickers for comparison
        
    Returns:
        Complete analysis with details, comparison, and explanations
    """
    print(f"Fetching stock details for {ticker}...")
    details = get_stock_details(ticker)
    
    print(f"Getting peer comparison...")
    comparison = get_peer_comparison(ticker, peers)
    
    print(f"Generating AI explanation for metrics...")
    metrics_explanation = explain_stock_metrics(details)
    
    print(f"Generating AI explanation for peer comparison...")
    peer_explanation = explain_peer_comparison(comparison)
    
    return {
        "ticker": ticker.upper(),
        "raw_details": details,
        "peer_comparison": comparison,
        "metrics_explanation": metrics_explanation,
        "peer_comparison_explanation": peer_explanation,
        "timestamp": details.get("timestamp", "N/A")
    }


# Example usage
if __name__ == "__main__":
    # Test with Apple
    ticker = "AAPL"
    
    # Test 1: Market News
    print(f"\n{'='*80}")
    print(f"📰 MARKET NEWS for {ticker}")
    print(f"{'='*80}\n")
    
    news_result = get_market_news(ticker, max_news=5)
    if "error" not in news_result or news_result.get("news_data"):
        print(f"Company: {news_result['company_name']}")
        print(f"Sector: {news_result['sector']} | Industry: {news_result['industry']}")
        print(f"Total News Articles: {news_result['total_news_count']}\n")
        
        print("="*80)
        print("📝 BASIC UNDERSTANDING")
        print("="*80)
        print(news_result['summary']['basic_understanding'])
        
        print("\n" + "="*80)
        print("🏢 COMPANY PROFILE")
        print("="*80)
        print(news_result['summary']['company_profile'])
        
        print("\n" + "="*80)
        print("📋 NEWS ARTICLES (JSON)")
        print("="*80)
        print(json.dumps(news_result['news_data'], indent=2))
    else:
        print(f"Error: {news_result.get('error', 'Unknown error')}")
    
    # Test 2: Complete Stock Analysis
    print(f"\n\n{'='*80}")
    print(f"📊 COMPLETE STOCK ANALYSIS for {ticker}")
    print(f"{'='*80}\n")
    
    analysis = get_complete_stock_analysis(ticker)
    
    print("\n" + "="*80)
    print("📊 STOCK METRICS EXPLANATION")
    print("="*80)
    print(analysis["metrics_explanation"])
    
    print("\n" + "="*80)
    print("📈 PEER COMPARISON EXPLANATION")
    print("="*80)
    print(analysis["peer_comparison_explanation"])
    
    print("\n" + "="*80)
    print("📋 RAW DATA")
    print("="*80)
    print(json.dumps(analysis["raw_details"], indent=2))

