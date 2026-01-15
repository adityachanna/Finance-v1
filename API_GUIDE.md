# Stock Details API - Complete Guide

## Overview
This API provides comprehensive stock analysis using Yahoo Finance data and AI-powered explanations for consumer-friendly understanding.

## Features

### 1. **Stock Details** (`get_stock_details`)
Fetches comprehensive financial metrics for any stock:
- **Price Information**: Current price, 52-week high/low, day change
- **Valuation Metrics**: PE ratio, Price-to-Book, Price-to-Sales, PEG ratio
- **Profitability**: Profit margin, Operating margin, ROE, ROA
- **Growth**: Revenue growth, Earnings growth, YoY price growth
- **Risk**: Beta (volatility measure)
- **Dividends**: Dividend yield, Payout ratio
- **Analyst Data**: Recommendations, Target price

### 2. **Market News** (`get_market_news`) ✨ NEW
Fetches latest market news and generates AI summaries:
- **News Data**: Title, Publisher, Link, Publish time, Thumbnail
- **AI Summary (2 parts)**:
  1. **Basic Understanding**: What's happening? Is news positive/negative?
  2. **Company Profile**: What does the company do? Market position?
- **Output Format**: Full JSON with structured data

### 3. **Peer Comparison** (`get_peer_comparison`)
Compares stock with industry peers on key metrics:
- PE ratio comparison
- Market cap comparison
- Profitability comparison (margins, ROE)
- Growth comparison
- Risk comparison (beta)

### 4. **AI Explanations**
- **Metrics Explanation** (`explain_stock_metrics`): Simple explanations of all metrics
- **Peer Comparison Explanation** (`explain_peer_comparison`): How stock compares to peers

### 5. **Complete Analysis** (`get_complete_stock_analysis`)
Master function combining all features above

---

## Usage Examples

### Example 1: Get Market News
```python
from details import get_market_news
import json

# Get news for Apple
news_data = get_market_news("AAPL", max_news=10)

# Access summary
print(news_data['summary']['basic_understanding'])
print(news_data['summary']['company_profile'])

# Access news articles
for article in news_data['news_data']:
    print(f"{article['title']} - {article['publisher']}")
    print(f"Link: {article['link']}")
```

### Example 2: Get Stock Details
```python
from details import get_stock_details

details = get_stock_details("MSFT")

print(f"PE Ratio: {details['pe_ratio_trailing']}")
print(f"Market Cap: ${details['market_cap']:,.0f}")
print(f"Profit Margin: {details['profit_margin']}%")
```

### Example 3: Complete Analysis
```python
from details import get_complete_stock_analysis

analysis = get_complete_stock_analysis("GOOGL")

# Get AI explanation
print(analysis['metrics_explanation'])
print(analysis['peer_comparison_explanation'])

# Get raw data
print(analysis['raw_details'])
```

---

## API Response Structures

### Market News Response
```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "sector": "Technology",
  "industry": "Consumer Electronics",
  "total_news_count": 10,
  "summary": {
    "basic_understanding": "AI-generated plain English summary...",
    "company_profile": "AI-generated company profile..."
  },
  "news_data": [
    {
      "title": "Article title",
      "publisher": "Reuters",
      "link": "https://...",
      "publish_time": 1642012800,
      "type": "STORY",
      "thumbnail": "https://...",
      "related_tickers": ["AAPL", "MSFT"]
    }
  ]
}
```

### Stock Details Response
```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "sector": "Technology",
  "current_price": 150.25,
  "pe_ratio_trailing": 28.5,
  "market_cap": 2500000000000,
  "profit_margin": 25.5,
  "revenue_growth": 8.6,
  "beta": 1.2,
  "dividend_yield": 0.52,
  ...
}
```

---

## Key Benefits for Consumers

1. **Plain English Explanations**: All complex metrics explained in simple terms
2. **News Context**: Stay updated with latest developments and their implications
3. **Peer Comparison**: Understand relative performance vs competitors
4. **Comprehensive Data**: All essential metrics in one API call
5. **AI-Powered Insights**: Smart summaries and analysis

---

## Free Data Source
All data comes from **Yahoo Finance** via the `yfinance` library - completely free!

---

## Dependencies
```bash
pip install yfinance langchain-groq python-dotenv
```

## Environment Setup
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Function Reference

| Function | Purpose | Returns |
|----------|---------|---------|
| `get_stock_details(ticker)` | Get financial metrics | Dict with stock data |
| `get_market_news(ticker, max_news)` | Get news + AI summary | Dict with news & summaries |
| `get_peer_comparison(ticker, peers)` | Compare with peers | Dict with comparison |
| `explain_stock_metrics(details)` | AI explanation of metrics | String (plain text) |
| `explain_peer_comparison(comparison)` | AI comparison explanation | String (plain text) |
| `get_complete_stock_analysis(ticker)` | Everything combined | Dict with all data |

---

## Notes
- All prices in company's reporting currency (usually USD)
- Percentages already formatted (multiply by 100 done)
- "N/A" returned when data unavailable
- News summaries are AI-generated - may vary
- Peer groups predefined for major stocks, defaulting to tech giants otherwise
